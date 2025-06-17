document.addEventListener('DOMContentLoaded', function() {
    const viewModeSelect = document.getElementById('viewModeSelect');
    const filterSelect = document.getElementById('filterSelect');
    const ctx = document.getElementById('resultsChart').getContext('2d');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const chartCanvas = document.getElementById('resultsChart');
    let chart;
    let rawData = [];
    let modelOrder = [];

    async function fetchData() {
        try {
            const response = await fetch('/api/results/v2');
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
        
        if (isModelView) {
            filterKeys.sort((a, b) => modelOrder.indexOf(a) - modelOrder.indexOf(b));
        }

        filterSelect.innerHTML = filterKeys.map(key => `<option value="${key}">${key}</option>`).join('');

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

    function renderChart() {
        const isModelView = viewModeSelect.value === 'model';
        const filterValue = filterSelect.value;
        const groupBy = isModelView ? 'model_name' : 'grammatical_phenomenon';
        const xAxisKey = isModelView ? 'grammatical_phenomenon' : 'model_name';

        const filteredData = rawData.filter(item => item[groupBy] === filterValue);

        if (xAxisKey === 'model_name') {
            filteredData.sort((a, b) => modelOrder.indexOf(a[xAxisKey]) - modelOrder.indexOf(b[xAxisKey]));
        }

        const labels = filteredData.map(item => {
            const mainLabel = item[xAxisKey];
            const language = item['dataset_language'];
            const shortLabel = mainLabel.length > 15 ? mainLabel.substring(0, 15) + '...' : mainLabel;
            return `${shortLabel} (${language})`;
        });

        const scatterData = filteredData.map((item, index) => ({
            x: index,
            y: item.accuracy,
            fullLabel: `${item[xAxisKey]} (${item['dataset_language']})`
        }));

        const trendlineData = calculateTrendline(scatterData);

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Accuracy',
                    data: scatterData,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 2,
                    pointRadius: 8,
                    pointHoverRadius: 12,
                    pointBackgroundColor: 'rgba(102, 126, 234, 0.9)',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointHoverBackgroundColor: 'rgba(102, 126, 234, 1)',
                    pointHoverBorderColor: '#ffffff',
                    pointHoverBorderWidth: 3
                }, {
                    label: 'Trend Line',
                    data: trendlineData,
                    type: 'line',
                    borderColor: 'rgba(231, 76, 60, 0.8)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    fill: false,
                    tension: 0
                }]
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
                        text: `Accuracy Analysis: ${filterValue}`,
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
                            title: function(context) {
                                if (context[0].datasetIndex === 0) {
                                    const index = context[0].dataIndex;
                                    return scatterData[index].fullLabel;
                                }
                                return '';
                            },
                            label: function(context) {
                                if (context.datasetIndex === 0) {
                                    const index = context.dataIndex;
                                    const dataPoint = filteredData[index];
                                    return [
                                        `Accuracy: ${context.parsed.y.toFixed(4)}`,
                                        `Perplexity (Good): ${dataPoint['perplexity_good'].toFixed(2)}`,
                                        `Perplexity (Bad): ${dataPoint['perplexity_bad'].toFixed(2)}`
                                    ];
                                }
                                return null;
                            }
                        }
                    }
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
                            text: isModelView ? 'Grammatical Phenomena' : 'Models (Dataset Language)',
                            font: {
                                size: 14,
                                weight: 'bold'
                            },
                            color: '#34495e'
                        },
                        ticks: {
                            values: Array.from({length: labels.length}, (_, i) => i),
                            callback: function(value) {
                                return labels[value];
                            },
                            font: {
                                size: 10
                            },
                            color: '#34495e',
                            maxRotation: 45,
                            minRotation: 45
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

    viewModeSelect.addEventListener('change', updateView);
    filterSelect.addEventListener('change', renderChart);

    fetchData();
});