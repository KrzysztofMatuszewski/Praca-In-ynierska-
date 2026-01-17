document.addEventListener('DOMContentLoaded', function() {
    // First Chart: Line Chart

    var ctx = document.getElementById("lineChart").getContext("2d");
    var lineChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Odsetek Anomali",
                    data: values,
                    fill: false,
                    borderColor: "rgb(75, 192, 192)",
                    lineTension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0
                }
            ]
        },
        options: {
            responsive: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Liczba Rekorów'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frakcja'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4) + ', (MSE: ' + mse[context.dataIndex].toFixed(4) + ')';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });


    var ctx = document.getElementById('featureImportanceChart').getContext('2d');
    var featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: feature,
            datasets: [{
                label: 'Feature Importance',
                data: importance,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            title: {
                display: true,
                text: 'Feature Importance'
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Features'
                    },
                    ticks: {
                        autoSkip: false,
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Importance'
                    }
                }
            }
        }
    });

    // Calculate the number of anomalies and non-anomalies
    var numAnomalies = anomalies.filter(value => value === 1).length;
    var numNonAnomalies = anomalies.length - numAnomalies;  // Total length minus the count of anomalies

    // Get the context of the canvas element
    var ctx = document.getElementById('anomalyDetectionSummary').getContext('2d');
    var anomalyDetectionSummary = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Anomalies', 'Non-Anomalies'],
            datasets: [{
                label: 'Count',
                data: [numAnomalies, numNonAnomalies],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)', // Color for anomalies
                    'rgba(54, 162, 235, 0.2)'  // Color for non-anomalies
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)', // Border color for anomalies
                    'rgba(54, 162, 235, 1)'  // Border color for non-anomalies
                ],
                borderWidth: 1
            }]
        },
        options: {
            title: {
                display: true,
                text: 'Anomaly Detection Summary'
            }
        }
    });

    function createFeatureData(data, anomaly_array_as_strings) {
        const booleanAnomalyArray = anomaly_array_as_strings.map(value => value === 'True');
        const normalData = data.filter((_, index) => !booleanAnomalyArray[index]);
        const anomalyData = data.filter((_, index) => booleanAnomalyArray[index]);
    
        const uniqueValues = [...new Set(data)].sort((a, b) => a - b);
        const normalCounts = uniqueValues.map(value => normalData.filter(v => v === value).length);
        const anomalyCounts = uniqueValues.map(value => anomalyData.filter(v => v === value).length);
    
        return { uniqueValues, normalCounts, anomalyCounts };
    }
    
    function createChart(canvas, chartType, labels, normalCounts, anomalyCounts, feature_name) {
        const datasets = [
            {
                label: 'Normal',
                data: normalCounts,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: 'Anomalies',
                data: anomalyCounts,
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ];
    
        const options = {
            plugins: {
                title: {
                    display: true,
                    text: `Distribution of ${feature_name}`
                }
            },
            responsive: true,
            maintainAspectRatio: false
        };
    
        if (chartType === 'bar') {
            options.scales = {
                x: {
                    title: {
                        display: true,
                        text: 'Feature Value'
                    },
                    stacked: true
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    },
                    beginAtZero: true,
                    stacked: true
                }
            };
        }
    
        new Chart(canvas, {
            type: chartType,
            data: {
                labels: labels,
                datasets: datasets
            },
            options: options
        });
    }
    
    function plotFeatureDistributions() {
        const container = document.getElementById('chartsContainer');
        container.innerHTML = '';
        container.style.display = 'flex';
        container.style.flexWrap = 'wrap';
        container.style.justifyContent = 'space-around';
    
        filtered_headers.forEach((feature_name, index) => {
            const { uniqueValues, normalCounts, anomalyCounts } = createFeatureData(column_arrays[feature_name], anomaly_array_as_strings);
    
            const chartContainer = document.createElement('div');
            chartContainer.style.width = '45%';
            chartContainer.style.height = '400px';
            chartContainer.style.margin = '10px';
            container.appendChild(chartContainer);
    
            const canvas = document.createElement('canvas');
            canvas.width = 900;
            canvas.height = 400;
            chartContainer.appendChild(canvas);
    
            const chartType = uniqueValues.length > 150 ? 'bubble' : 'bar';
            createChart(canvas, chartType, uniqueValues, normalCounts, anomalyCounts, feature_name);
    
            // Add a line break after every two charts
            if (index % 2 !== 0) {
                const lineBreak = document.createElement('div');
                lineBreak.style.flexBasis = '100%';
                lineBreak.style.height = '0';
                container.appendChild(lineBreak);
            }
        });
    }
    
    plotFeatureDistributions();
});