document.addEventListener('DOMContentLoaded', function() {
    const viewModeSelect = document.getElementById('viewModeSelect');
    const filterSelect = document.getElementById('filterSelect');
    const ctx = document.getElementById('resultsChart').getContext('2d');
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
            updateView();
        } catch (error) {
            console.error('Error fetching data:', error);
            alert('Error fetching data: ' + error.message);
        }
    }

    function updateView() {
        const isModelView = viewModeSelect.value === 'model';
        const groupBy = isModelView ? 'model_name' : 'grammatical_phenomenon';

        const filterKeys = [...new Set(rawData.map(item => item[groupBy]))];
        
        if (isModelView) {
            // Sort models based on the provided order
            filterKeys.sort((a, b) => modelOrder.indexOf(a) - modelOrder.indexOf(b));
        }

        filterSelect.innerHTML = filterKeys.map(key => `<option value="${key}">${key}</option>`).join('');

        renderChart();
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

        const labels = filteredData.map(item => `${item[xAxisKey]} (${item['dataset_language']})`);
        const accuracyData = filteredData.map(item => item.accuracy);

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Accuracy',
                    data: accuracyData,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                animation: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Accuracy'
                        }
                    },
                    x: {
                         title: {
                            display: true,
                            text: isModelView ? 'Grammatical Phenomenon' : 'Model Name'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const index = context.dataIndex;
                                const dataPoint = filteredData[index];
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(4);
                                }
                                return [
                                    label,
                                    `Perplexity (Good): ${dataPoint['perplexity_good'].toFixed(2)}`,
                                    `Perplexity (Bad): ${dataPoint['perplexity_bad'].toFixed(2)}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }

    viewModeSelect.addEventListener('change', updateView);
    filterSelect.addEventListener('change', renderChart);

    fetchData();
});
