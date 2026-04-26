document.addEventListener('DOMContentLoaded', function () {
    const viewModeSelect = document.getElementById('viewModeSelect');
    const filterSelect = document.getElementById('filterSelect');
    const clearFiltersBtn = document.getElementById('clearFilters');
    const showTrendLineToggle = document.getElementById('showTrendLine');
    const ctx = document.getElementById('resultsChart').getContext('2d');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const chartCanvas = document.getElementById('resultsChart');

    let chart;
    let rawData = [];
    let modelOrder = [];
    let trendlineDataset = null;

    // Color palette for different series
    const colorPalette = [
        '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ];

    async function fetchData() {
        try {
            const response = await fetch('/api/results');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to fetch data');
            }
            const data = await response.json();
            rawData = data.results;
            modelOrder = data.model_order;

            loadingIndicator.style.display = 'none';
            chartCanvas.style.display = 'block';
            updateView();
        } catch (error) {
            console.error('Error fetching data:', error);
            loadingIndicator.innerHTML = '<div style="color: #e74c3c; text-align: center;">Error loading data: ' + error.message + '</div>';
        }
    }

    function updateView() {
        const isModelView = viewModeSelect.value === 'model';
        const groupBy = isModelView ? 'model_name' : 'grammatical_phenomenon';
        const filterKeys = [...new Set(rawData.map(item => item[groupBy]))];

        // Save current selections
        const currentSelections = Array.from(filterSelect.selectedOptions).map(opt => opt.value);

        // Clear and repopulate the filter select
        filterSelect.innerHTML = '';

        // Sort filter keys appropriately
        if (isModelView) {
            filterKeys.sort((a, b) => modelOrder.indexOf(a) - modelOrder.indexOf(b));
        } else {
            filterKeys.sort();
        }

        // Add options to the select
        filterKeys.forEach(key => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = key;
            option.selected = currentSelections.includes(key);
            filterSelect.appendChild(option);
        });

        // If no selections and there are options, select the first one
        if (filterSelect.selectedOptions.length === 0 && filterKeys.length > 0) {
            filterSelect.options[0].selected = true;
        }

        renderChart();
    }

    function calculateTrendline(data) {
        const n = data.length;
        const xValues = data.map((_, index) => index);
        const yValues = data.map(point => point.y);

        const sumX = xValues.reduce((sum, x) => sum + x, 0);
        const sumY = yValues.reduce((sum, y) => sum + y, 0);
        const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
        const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        return xValues.map(x => ({
            x: x,
            y: slope * x + intercept
        }));
    }

    function getGroupedData() {
        const isModelView = viewModeSelect.value === 'model';
        const groupBy = isModelView ? 'model_name' : 'grammatical_phenomenon';
        const xAxisKey = isModelView ? 'grammatical_phenomenon' : 'model_name';

        // Get selected filters
        const selectedFilters = Array.from(filterSelect.selectedOptions).map(opt => opt.value);
        if (selectedFilters.length === 0) return { datasets: [], labels: [] };

        // Filter data based on selected filters
        const filteredData = rawData.filter(item => selectedFilters.includes(item[groupBy]));

        let allXValues = [...new Set(filteredData.map(item => item[xAxisKey]))];

        if (isModelView) {
            allXValues.sort(); // Sort alphabetically for model view
        } else {
            // In phenomenon view, sort models by model order
            allXValues.sort((a, b) => modelOrder.indexOf(a) - modelOrder.indexOf(b));
        }

        // Create x-axis labels with language info
        const labels = allXValues;

        // Group data by the selected filter groups
        const groupedData = new Map();

        selectedFilters.forEach((filter, index) => {
            const groupData = filteredData.filter(item => item[groupBy] === filter);
            const color = colorPalette[index % colorPalette.length];

            // Map each x-value to its corresponding data point
            const points = allXValues.map(xValue => {
                const point = groupData.find(item => item[xAxisKey] === xValue);
                if (!point) return null;

                return {
                    x: allXValues.indexOf(xValue),
                    y: point.accuracy,
                    fullLabel: `${point[xAxisKey]} (${point.dataset_language})`,
                    perplexity_good: point.perplexity_good,
                    perplexity_bad: point.perplexity_bad
                };
            }).filter(Boolean);

            if (points.length > 0) {
                groupedData.set(filter, {
                    label: filter,
                    data: points,
                    color: color,
                    borderColor: color,
                    backgroundColor: color + '80', // Add alpha for fill
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 12,
                    pointBackgroundColor: color + 'CC',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointHoverBackgroundColor: color,
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 3,
                    showLine: true,
                    tension: 0.1
                });
            }
        });

        return {
            datasets: Array.from(groupedData.values()),
            labels: labels
        };
    }

    function renderChart() {
        const { datasets, labels } = getGroupedData();
        const isModelView = viewModeSelect.value === 'model';

        if (chart) {
            chart.destroy();
        }

        // Calculate trendline for each dataset if enabled
        const trendlineDatasets = [];
        if (showTrendLineToggle.checked) {
            datasets.forEach(dataset => {
                const trendlineData = calculateTrendline(dataset.data);
                trendlineDatasets.push({
                    label: `${dataset.label} - Trend`,
                    data: trendlineData,
                    type: 'line',
                    borderColor: dataset.borderColor,
                    backgroundColor: dataset.backgroundColor,
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false,
                    tension: 0.1
                });
            });
        }

        chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [...datasets, ...trendlineDatasets]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        top: 10,
                        bottom: 20,
                        left: 20,
                        right: 20
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeInOutQuart'
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Accuracy Analysis: ${viewModeSelect.value === 'model' ? 'Grouped by Model' : 'Grouped by Phenomenon'}`,
                        font: {
                            size: 20,
                            weight: 'bold'
                        },
                        color: '#2c3e50',
                        padding: {
                            top: 5,
                            bottom: 20
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: {
                                size: 12,
                                weight: '600'
                            },
                            color: '#34495e',
                            padding: 15,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(44, 62, 80, 0.95)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2,
                        cornerRadius: 8,
                        titleFont: {
                            size: 12,
                            weight: 'bold'
                        },
                        bodyFont: {
                            size: 11
                        },
                        callbacks: {
                            title: function (context) {
                                const datasetIndex = context[0].datasetIndex;
                                const dataIndex = context[0].dataIndex;
                                const dataset = chart.data.datasets[datasetIndex];

                                // Skip trend lines in title
                                if (dataset.label && dataset.label.endsWith('Trend')) {
                                    return '';
                                }

                                // Get the data point
                                const dataPoint = dataset.data[dataIndex];
                                return dataPoint.fullLabel || '';
                            },
                            label: function (context) {
                                const datasetIndex = context.datasetIndex;
                                const dataset = chart.data.datasets[datasetIndex];
                                const dataPoint = dataset.data[context.dataIndex];

                                // Handle trend lines
                                if (dataset.label && dataset.label.endsWith('Trend')) {
                                    return [
                                        `${dataset.label}`,
                                        `Trend value: ${context.parsed.y.toFixed(4)}`
                                    ];
                                }

                                // Handle regular data points
                                return [
                                    `Accuracy: ${context.parsed.y.toFixed(4)}`,
                                    `Perplexity (Good): ${dataPoint.perplexity_good ? dataPoint.perplexity_good.toFixed(2) : 'N/A'}`,
                                    `Perplexity (Bad): ${dataPoint.perplexity_bad ? dataPoint.perplexity_bad.toFixed(2) : 'N/A'}`,
                                    `Model: ${dataset.label}`
                                ];
                            }
                        }
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Accuracy Score',
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#34495e'
                        },
                        ticks: {
                            font: {
                                size: 12
                            },
                            color: '#34495e'
                        },
                        grid: {
                            color: 'rgba(52, 73, 94, 0.1)',
                            lineWidth: 1
                        }
                    },
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: labels.length - 0.5,
                        title: {
                            display: true,
                            text: isModelView ? 'Grammatical Phenomena' : 'Models',
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#34495e'
                        },
                        ticks: {
                            values: Array.from({ length: labels.length }, (_, i) => i),
                            callback: function (value) {
                                return labels[value];
                            },
                            font: {
                                size: 16
                            },
                            color: '#34495e',
                            maxRotation: 30,
                            minRotation: 30
                        },
                        grid: {
                            color: 'rgba(52, 73, 94, 0.1)',
                            lineWidth: 1
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'point'
                }
            }
        });
    }


    // Add event listeners
    viewModeSelect.addEventListener('change', () => {
        updateView();
    });


    // Clear all selected filters
    clearFiltersBtn.addEventListener('click', () => {
        Array.from(filterSelect.options).forEach(option => {
            option.selected = false;
        });
        renderChart();
    });

    // Prevent closing the dropdown when clicking inside
    filterSelect.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', () => {
        filterSelect.size = 8;
    });

    // Toggle dropdown size on focus/blur
    filterSelect.addEventListener('focus', () => {
        filterSelect.size = 8;
    });

    filterSelect.addEventListener('blur', () => {
        // Small delay to allow for selection
        setTimeout(() => {
            filterSelect.size = 8;
        }, 200);
    });

    viewModeSelect.addEventListener('change', updateView);
    filterSelect.addEventListener('change', renderChart);
    showTrendLineToggle.addEventListener('change', renderChart);

    fetchData();

    function toggleTrendLine() {
        renderChart();
    }
});