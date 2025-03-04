
// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

/**
 * Initialize the dashboard
 */
function initializeDashboard() {
    // Initialize theme
    initializeTheme();
    
    // Initialize navigation
    initializeNavigation();
    
    // Initialize data loading
    loadModels();
    loadResults();
    
    // Initialize event listeners
    initializeEventListeners();
}

/**
 * Initialize theme handling
 */
function initializeTheme() {
    // Check if user has a saved theme preference
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else {
        // Check for system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.documentElement.setAttribute('data-theme', 'dark');
        }
    }
    
    // Add event listeners for theme toggle
    const themeOptions = document.querySelectorAll('[data-theme]');
    themeOptions.forEach(option => {
        option.addEventListener('click', function(e) {
            e.preventDefault();
            const theme = this.getAttribute('data-theme');
            
            if (theme === 'auto') {
                localStorage.removeItem('theme');
                if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    document.documentElement.setAttribute('data-theme', 'dark');
                } else {
                    document.documentElement.setAttribute('data-theme', 'light');
                }
            } else {
                localStorage.setItem('theme', theme);
                document.documentElement.setAttribute('data-theme', theme);
            }
            
            // Update Plotly charts for theme
            updateChartsForTheme();
        });
    });
}

/**
 * Initialize navigation between views
 */
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-view]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the view to show
            const viewId = this.getAttribute('data-view');
            
            // Update active link
            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
            
            // Hide all views and show the selected one
            const views = document.querySelectorAll('.view-section');
            views.forEach(view => {
                view.classList.remove('active');
            });
            
            const selectedView = document.getElementById(viewId);
            if (selectedView) {
                selectedView.classList.add('active');
                
                // Update the view title
                const viewTitle = this.textContent.trim();
                document.getElementById('current-view-title').textContent = viewTitle;
            }
            
            // On mobile, collapse the sidebar after navigation
            if (window.innerWidth < 768) {
                document.getElementById('sidebar').classList.remove('show');
            }
        });
    });
}

/**
 * Load available models from the API
 */
function loadModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            populateModelSelector(data);
        })
        .catch(error => {
            console.error('Error loading models:', error);
        });
}

/**
 * Populate the model selector dropdown
 */
function populateModelSelector(models) {
    const selector = document.getElementById('model-selector');
    
    // Clear existing options
    while (selector.options.length > 1) {
        selector.remove(1);
    }
    
    // Add models to selector
    for (const [modelName, modelConfig] of Object.entries(models)) {
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        selector.appendChild(option);
    }
}

/**
 * Load available benchmark results from the API
 */
function loadResults() {
    fetch('/api/results')
        .then(response => response.json())
        .then(data => {
            populateResultSelector(data);
            populateComparisonLists(data);
        })
        .catch(error => {
            console.error('Error loading results:', error);
        });
}

/**
 * Populate the result selector dropdown
 */
function populateResultSelector(results) {
    const selector = document.getElementById('result-selector');
    
    // Clear existing options
    while (selector.options.length > 1) {
        selector.remove(1);
    }
    
    // Add results to selector
    for (const [resultId, resultData] of Object.entries(results)) {
        const option = document.createElement('option');
        option.value = resultId;
        
        // Create a descriptive label
        let label = resultId;
        if (resultData.model_name) {
            label = `${resultData.model_name} - ${resultId}`;
        }
        if (resultData.timestamp) {
            const date = new Date(resultData.timestamp * 1000);
            label += ` (${date.toLocaleString()})`;
        }
        
        option.textContent = label;
        selector.appendChild(option);
    }
}

/**
 * Populate the comparison lists
 */
function populateComparisonLists(results) {
    const availableResults = document.getElementById('available-results');
    
    // Clear existing options
    while (availableResults.options.length > 0) {
        availableResults.remove(0);
    }
    
    // Add results to available list
    for (const [resultId, resultData] of Object.entries(results)) {
        const option = document.createElement('option');
        option.value = resultId;
        
        // Create a descriptive label
        let label = resultId;
        if (resultData.model_name) {
            label = `${resultData.model_name} - ${resultId}`;
        }
        if (resultData.timestamp) {
            const date = new Date(resultData.timestamp * 1000);
            label += ` (${date.toLocaleString()})`;
        }
        
        option.textContent = label;
        availableResults.appendChild(option);
    }
}

/**
 * Initialize event listeners for dashboard interactions
 */
function initializeEventListeners() {
    // Result selector change
    document.getElementById('result-selector').addEventListener('change', function() {
        const resultId = this.value;
        if (resultId && resultId !== 'Select a benchmark result...') {
            loadResultData(resultId);
        }
    });
    
    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', function() {
        // Reload current data
        loadModels();
        loadResults();
        
        // If a result is selected, reload it
        const resultId = document.getElementById('result-selector').value;
        if (resultId && resultId !== 'Select a benchmark result...') {
            loadResultData(resultId);
        }
    });
    
    // Export button
    document.getElementById('export-btn').addEventListener('click', function() {
        // Export current view as PNG
        exportCurrentView();
    });
    
    // Upload form
    document.getElementById('upload-button').addEventListener('click', function() {
        uploadResultFile();
    });
    
    // Comparison controls
    document.getElementById('add-comparison').addEventListener('click', function() {
        moveComparisonItems('available-results', 'comparison-results');
    });
    
    document.getElementById('remove-comparison').addEventListener('click', function() {
        moveComparisonItems('comparison-results', 'available-results');
    });
    
    document.getElementById('run-comparison').addEventListener('click', function() {
        runComparison();
    });
    
    // Live monitoring controls
    document.getElementById('connect-monitoring').addEventListener('click', function() {
        connectToMonitoring();
    });
    
    // Timeline controls
    document.getElementById('timeline-zoom-in').addEventListener('click', function() {
        zoomTimeline('in');
    });
    
    document.getElementById('timeline-zoom-out').addEventListener('click', function() {
        zoomTimeline('out');
    });
    
    document.getElementById('timeline-reset').addEventListener('click', function() {
        resetTimelineView();
    });
    
    document.getElementById('timeline-export').addEventListener('click', function() {
        exportTimelineAsPNG();
    });
}

/**
 * Load result data for a specific result ID
 */
function loadResultData(resultId) {
    fetch(`/api/profile/${resultId}`)
        .then(response => response.json())
        .then(data => {
            // Update UI with result data
            updateDashboardWithResultData(resultId, data);
            
            // Load recommendations for this result
            loadRecommendations(resultId);
        })
        .catch(error => {
            console.error(`Error loading result data for ${resultId}:`, error);
        });
}

/**
 * Update dashboard with result data
 */
function updateDashboardWithResultData(resultId, data) {
    // Update summary metrics
    document.getElementById('avg-latency').textContent = `${data.avg_latency.toFixed(2)} ms`;
    document.getElementById('throughput').textContent = `${data.throughput.toFixed(2)} samples/sec`;
    document.getElementById('peak-memory').textContent = `${(data.peak_memory / (1024 * 1024)).toFixed(2)} MB`;
    
    // Create visualizations
    createTimelineVisualization(data);
    createOperationBreakdown(data);
    createMemoryUsagePlot(data);
    createKernelEfficiencyPlot(data);
    
    // Create interactive operation explorer
    createOperationExplorer(data);
    
    // Create bottleneck visualization
    if (data.bottlenecks) {
        createBottleneckVisualization(data);
    }
    
    // Create parameter sensitivity plot
    if (data.sensitivity_analysis) {
        createParameterSensitivityPlot(data.sensitivity_analysis);
    }
}

/**
 * Load recommendations for a specific result
 */
function loadRecommendations(resultId) {
    fetch(`/api/recommendations/${resultId}`)
        .then(response => response.json())
        .then(data => {
            // Update UI with recommendations
            updateRecommendations(data);
            
            // Update optimization potential
            updateOptimizationPotential(data);
        })
        .catch(error => {
            console.error(`Error loading recommendations for ${resultId}:`, error);
        });
}

/**
 * Update recommendations display
 */
function updateRecommendations(recommendations) {
    const container = document.getElementById('top-recommendations-container');
    container.innerHTML = '';
    
    const recommendationsView = document.getElementById('recommendations-container');
    recommendationsView.innerHTML = '';
    
    if (recommendations.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No recommendations available</p></div>';
        recommendationsView.innerHTML = '<div class="text-center py-4"><p class="text-muted">No recommendations available</p></div>';
        return;
    }
    
    // Create recommendations for dashboard view (top 3)
    const topRecommendations = recommendations.slice(0, 3);
    
    topRecommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'card mb-3';
        
        card.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h6 class="m-0 font-weight-bold">${rec.title}</h6>
                <span class="badge bg-primary">${rec.type}</span>
            </div>
            <div class="card-body">
                <p>${rec.description}</p>
                <div class="d-flex justify-content-between">
                    <span>Latency: <strong class="text-success">-${rec.latency_improvement.toFixed(1)}%</strong></span>
                    <span>Throughput: <strong class="text-success">+${rec.throughput_improvement.toFixed(1)}%</strong></span>
                    <span>Memory: <strong class="text-success">-${rec.memory_improvement.toFixed(1)}%</strong></span>
                </div>
            </div>
        `;
        
        container.appendChild(card);
    });
    
    // Create full recommendations for recommendations view
    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'card mb-3';
        
        card.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="m-0 font-weight-bold">${rec.title}</h5>
                <span class="badge bg-primary">${rec.type}</span>
            </div>
            <div class="card-body">
                <p>${rec.description}</p>
                <div class="row mb-3">
                    <div class="col-md-4">
                        <div class="progress-indicator">
                            <div class="progress">
                                <div class="progress-bar bg-success" role="progressbar" style="width: ${rec.latency_improvement}%" 
                                    aria-valuenow="${rec.latency_improvement}" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                            <div class="progress-value">-${rec.latency_improvement.toFixed(1)}%</div>
                        </div>
                        <div class="text-center">Latency Improvement</div>
                    </div>
                    <div class="col-md-4">
                        <div class="progress-indicator">
                            <div class="progress">
                                <div class="progress-bar bg-info" role="progressbar" style="width: ${rec.throughput_improvement}%" 
                                    aria-valuenow="${rec.throughput_improvement}" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                            <div class="progress-value">+${rec.throughput_improvement.toFixed(1)}%</div>
                        </div>
                        <div class="text-center">Throughput Improvement</div>
                    </div>
                    <div class="col-md-4">
                        <div class="progress-indicator">
                            <div class="progress">
                                <div class="progress-bar bg-warning" role="progressbar" style="width: ${rec.memory_improvement}%" 
                                    aria-valuenow="${rec.memory_improvement}" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                            <div class="progress-value">-${rec.memory_improvement.toFixed(1)}%</div>
                        </div>
                        <div class="text-center">Memory Reduction</div>
                    </div>
                </div>
                <div class="implementation-details">
                    <h6>Implementation Details:</h6>
                    <p>${rec.implementation}</p>
                </div>
                <div class="d-flex justify-content-end">
                    <span class="text-muted">Confidence: ${(rec.confidence * 100).toFixed(0)}%</span>
                </div>
            </div>
        `;
        
        recommendationsView.appendChild(card);
    });
}

/**
 * Update optimization potential display
 */
function updateOptimizationPotential(recommendations) {
    if (recommendations.length === 0) {
        document.getElementById('optimization-potential').textContent = 'N/A';
        return;
    }
    
    // Calculate average improvement potential from top recommendations
    const topRecommendations = recommendations.slice(0, 3);
    
    const avgLatency = topRecommendations.reduce((sum, rec) => sum + rec.latency_improvement, 0) / topRecommendations.length;
    const avgThroughput = topRecommendations.reduce((sum, rec) => sum + rec.throughput_improvement, 0) / topRecommendations.length;
    
    // Use the better of latency reduction or throughput improvement
    const potential = Math.max(avgLatency, avgThroughput);
    
    document.getElementById('optimization-potential').textContent = `${potential.toFixed(1)}%`;
}

/**
 * Create timeline visualization
 */
function createTimelineVisualization(data) {
    const container = document.getElementById('timeline-container');
    container.innerHTML = '';
    
    if (!data.operations || data.operations.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No timeline data available</p></div>';
        return;
    }
    
    // Extract operation data
    const operations = data.operations;
    
    // Sort operations by start time
    operations.sort((a, b) => a.start_time - b.start_time);
    
    // Extract operation data
    const opNames = [];
    const startTimes = [];
    const endTimes = [];
    const categories = [];
    const deviceTypes = [];
    
    operations.forEach(op => {
        opNames.push(op.name || 'Unknown');
        startTimes.push(op.start_time || 0);
        endTimes.push(op.end_time || 0);
        categories.push(op.category || 'Other');
        deviceTypes.push(op.device || 'CPU');
    });
    
    // Create a color map for operation categories
    const uniqueCategories = [...new Set(categories)];
    const colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];
    
    const colorMap = {};
    uniqueCategories.forEach((cat, i) => {
        colorMap[cat] = colors[i % colors.length];
    });
    
    // Create figure data
    const traces = [];
    
    // Add operation bars
    operations.forEach((op, i) => {
        const duration = endTimes[i] - startTimes[i];
        
        traces.push({
            x: [startTimes[i], endTimes[i]],
            y: [opNames[i], opNames[i]],
            mode: 'lines',
            line: {
                color: colorMap[categories[i]],
                width: 20
            },
            name: categories[i],
            text: `Duration: ${duration.toFixed(3)} ms<br>Device: ${deviceTypes[i]}`,
            hoverinfo: 'text',
            showlegend: false
        });
    });
    
    // Add legend traces
    uniqueCategories.forEach(cat => {
        traces.push({
            x: [null],
            y: [null],
            mode: 'lines',
            line: {
                color: colorMap[cat],
                width: 10
            },
            name: cat
        });
    });
    
    // Create layout
    const layout = {
        title: 'Operation Timeline',
        xaxis: {
            title: 'Time (ms)'
        },
        yaxis: {
            title: 'Operations'
        },
        margin: {
            l: 150,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest',
        legend: {
            orientation: 'h',
            y: -0.2
        },
        height: 500
    };
    
    // Create the plot
    Plotly.newPlot(container, traces, layout);
}

/**
 * Create operation breakdown visualization
 */
function createOperationBreakdown(data) {
    const container = document.getElementById('operation-breakdown-container');
    container.innerHTML = '';
    
    if (!data.operations || data.operations.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No operation data available</p></div>';
        return;
    }
    
    // Group operations by category
    const categories = {};
    data.operations.forEach(op => {
        const cat = op.category || 'Other';
        const duration = (op.end_time || 0) - (op.start_time || 0);
        
        if (cat in categories) {
            categories[cat] += duration;
        } else {
            categories[cat] = duration;
        }
    });
    
    // Convert to arrays for plotting
    const catNames = Object.keys(categories);
    const catDurations = Object.values(categories);
    
    // Calculate percentages
    const totalTime = catDurations.reduce((a, b) => a + b, 0);
    const percentages = catDurations.map(d => (d / totalTime * 100).toFixed(1));
    
    // Sort by duration
    const indices = catDurations.map((_, i) => i);
    indices.sort((a, b) => catDurations[b] - catDurations[a]);
    
    const sortedNames = indices.map(i => catNames[i]);
    const sortedDurations = indices.map(i => catDurations[i]);
    const sortedPercentages = indices.map(i => percentages[i]);
    
    // Create the trace
    const trace = {
        x: sortedNames,
        y: sortedDurations,
        type: 'bar',
        marker: {
            color: '#4e73df'
        },
        text: sortedPercentages.map(p => `${p}%`),
        textposition: 'auto',
        hovertemplate: '%{x}<br>Duration: %{y:.3f} ms<br>Percentage: %{text}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Operation Breakdown by Category',
        xaxis: {
            title: 'Category'
        },
        yaxis: {
            title: 'Duration (ms)'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 100
        },
        hovermode: 'closest'
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Create memory usage plot
 */
function createMemoryUsagePlot(data) {
    const container = document.getElementById('memory-usage-container');
    container.innerHTML = '';
    
    if (!data.memory_timeline || data.memory_timeline.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No memory data available</p></div>';
        return;
    }
    
    // Extract memory timeline data
    const timestamps = [];
    const memoryUsage = [];
    
    data.memory_timeline.forEach(entry => {
        timestamps.push(entry.timestamp || 0);
        memoryUsage.push((entry.memory_usage || 0) / (1024 * 1024)); // Convert to MB
    });
    
    // Find peak memory usage
    const peakMemory = Math.max(...memoryUsage);
    const peakIndex = memoryUsage.indexOf(peakMemory);
    
    // Create memory usage trace
    const memoryTrace = {
        x: timestamps,
        y: memoryUsage,
        mode: 'lines',
        fill: 'tozeroy',
        name: 'Memory Usage',
        line: {
            color: '#4e73df',
            width: 2
        },
        fillcolor: 'rgba(78, 115, 223, 0.1)',
        hovertemplate: 'Time: %{x:.3f} ms<br>Memory: %{y:.2f} MB<extra></extra>'
    };
    
    // Create peak memory marker
    const peakTrace = {
        x: [timestamps[peakIndex]],
        y: [peakMemory],
        mode: 'markers+text',
        marker: {
            color: '#e74a3b',
            size: 10
        },
        text: ['Peak'],
        textposition: 'top center',
        name: 'Peak Memory',
        hovertemplate: 'Time: %{x:.3f} ms<br>Peak Memory: %{y:.2f} MB<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Memory Usage Timeline',
        xaxis: {
            title: 'Time (ms)'
        },
        yaxis: {
            title: 'Memory Usage (MB)'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest',
        showlegend: true
    };
    
    // Create the plot
    Plotly.newPlot(container, [memoryTrace, peakTrace], layout);
}

/**
 * Create kernel efficiency plot
 */
function createKernelEfficiencyPlot(data) {
    const container = document.getElementById('kernel-efficiency-container');
    container.innerHTML = '';
    
    if (!data.kernels || data.kernels.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No kernel data available</p></div>';
        return;
    }
    
    // Extract kernel data
    const kernelNames = [];
    const efficiency = [];
    
    data.kernels.forEach(kernel => {
        kernelNames.push(kernel.name || 'Unknown');
        
        const theoretical = kernel.theoretical_flops || 1;
        const achieved = kernel.achieved_flops || 0;
        const eff = (achieved / theoretical * 100).toFixed(1);
        
        efficiency.push(parseFloat(eff));
    });
    
    // Sort by efficiency
    const indices = efficiency.map((_, i) => i);
    indices.sort((a, b) => efficiency[b] - efficiency[a]);
    
    const sortedNames = indices.map(i => kernelNames[i]);
    const sortedEfficiency = indices.map(i => efficiency[i]);
    
    // Create the trace
    const trace = {
        x: sortedNames.slice(0, 10), // Show top 10 kernels
        y: sortedEfficiency.slice(0, 10),
        type: 'bar',
        marker: {
            color: sortedEfficiency.slice(0, 10),
            colorscale: [
                [0, '#e74a3b'],
                [0.5, '#f6c23e'],
                [1, '#1cc88a']
            ],
            cmin: 0,
            cmax: 100
        },
        text: sortedEfficiency.slice(0, 10).map(e => `${e}%`),
        textposition: 'auto',
        hovertemplate: '%{x}<br>Efficiency: %{text}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Top 10 Kernels by Efficiency',
        xaxis: {
            title: 'Kernel',
            tickangle: -45
        },
        yaxis: {
            title: 'Efficiency (%)',
            range: [0, 100]
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 100
        },
        hovermode: 'closest'
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Create interactive operation explorer
 */
function createOperationExplorer(data) {
    const container = document.getElementById('operation-explorer-container');
    container.innerHTML = '';
    
    if (!data.operations || data.operations.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No operation data available</p></div>';
        return;
    }
    
    // Sort operations by duration
    const operations = [...data.operations];
    operations.sort((a, b) => {
        const aDuration = (a.end_time || 0) - (a.start_time || 0);
        const bDuration = (b.end_time || 0) - (b.start_time || 0);
        return bDuration - aDuration;
    });
    
    // Create the explorer UI
    const explorerHTML = `
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="mb-3">
                    <label for="category-filter" class="form-label">Filter by Category</label>
                    <select class="form-select" id="category-filter">
                        <option value="all" selected>All Categories</option>
                        ${[...new Set(operations.map(op => op.category || 'Other'))].map(cat => 
                            `<option value="${cat}">${cat}</option>`
                        ).join('')}
                    </select>
                </div>
            </div>
            <div class="col-md-6">
                <div class="mb-3">
                    <label for="operation-search" class="form-label">Search Operations</label>
                    <input type="text" class="form-control" id="operation-search" placeholder="Search...">
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="operation-explorer" id="operation-list">
                    ${operations.map((op, index) => {
                        const duration = (op.end_time || 0) - (op.start_time || 0);
                        return `
                            <div class="operation-row" data-index="${index}" data-category="${op.category || 'Other'}">
                                <div class="d-flex justify-content-between">
                                    <strong>${op.name || 'Unknown'}</strong>
                                    <span class="badge ${getBadgeClass(op.device || 'CPU')}">${op.device || 'CPU'}</span>
                                </div>
                                <div class="d-flex justify-content-between mt-1">
                                    <small class="text-muted">${op.category || 'Other'}</small>
                                    <small>${duration.toFixed(3)} ms</small>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
            <div class="col-md-6">
                <div class="operation-details" id="operation-details">
                    <div class="text-center py-4">
                        <p class="text-muted">Select an operation to view details</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = explorerHTML;
    
    // Add event listeners
    document.getElementById('category-filter').addEventListener('change', function() {
        filterOperations();
    });
    
    document.getElementById('operation-search').addEventListener('input', function() {
        filterOperations();
    });
    
    document.querySelectorAll('.operation-row').forEach(row => {
        row.addEventListener('click', function() {
            // Remove active class from all rows
            document.querySelectorAll('.operation-row').forEach(r => {
                r.classList.remove('active');
            });
            
            // Add active class to clicked row
            this.classList.add('active');
            
            // Show operation details
            const index = parseInt(this.getAttribute('data-index'));
            showOperationDetails(operations[index]);
        });
    });
    
    // Function to filter operations
    function filterOperations() {
        const category = document.getElementById('category-filter').value;
        const search = document.getElementById('operation-search').value.toLowerCase();
        
        document.querySelectorAll('.operation-row').forEach(row => {
            const rowCategory = row.getAttribute('data-category');
            const rowText = row.textContent.toLowerCase();
            
            const categoryMatch = category === 'all' || rowCategory === category;
            const searchMatch = search === '' || rowText.includes(search);
            
            if (categoryMatch && searchMatch) {
                row.style.display = 'block';
            } else {
                row.style.display = 'none';
            }
        });
    }
    
    // Function to show operation details
    function showOperationDetails(operation) {
        const detailsContainer = document.getElementById('operation-details');
        
        const duration = (operation.end_time || 0) - (operation.start_time || 0);
        const memoryUsed = ((operation.memory_used || 0) / (1024 * 1024)).toFixed(2); // Convert to MB
        
        let inputShapes = 'N/A';
        if (operation.input_shapes && operation.input_shapes.length > 0) {
            inputShapes = operation.input_shapes.map(shape => shape.join('×')).join(', ');
        }
        
        let outputShapes = 'N/A';
        if (operation.output_shapes && operation.output_shapes.length > 0) {
            outputShapes = operation.output_shapes.map(shape => shape.join('×')).join(', ');
        }
        
        detailsContainer.innerHTML = `
            <h5>${operation.name || 'Unknown'}</h5>
            <p><strong>Category:</strong> ${operation.category || 'Other'}</p>
            <p><strong>Device:</strong> ${operation.device || 'CPU'}</p>
            <p><strong>Duration:</strong> ${duration.toFixed(3)} ms</p>
            <p><strong>Memory Used:</strong> ${memoryUsed} MB</p>
            <p><strong>FLOPS:</strong> ${operation.flops ? operation.flops.toExponential(2) : 'N/A'}</p>
            <p><strong>Input Shapes:</strong> ${inputShapes}</p>
            <p><strong>Output Shapes:</strong> ${outputShapes}</p>
            <p><strong>Start Time:</strong> ${operation.start_time.toFixed(3)} ms</p>
            <p><strong>End Time:</strong> ${operation.end_time.toFixed(3)} ms</p>
        `;
    }
    
    // Function to get badge class based on device
    function getBadgeClass(device) {
        switch (device.toLowerCase()) {
            case 'gpu':
                return 'bg-success';
            case 'cpu':
                return 'bg-primary';
            case 'tpu':
                return 'bg-warning';
            default:
                return 'bg-secondary';
        }
    }
}

/**
 * Create bottleneck visualization
 */
function createBottleneckVisualization(data) {
    const container = document.getElementById('bottleneck-visualization-container');
    container.innerHTML = '';
    
    if (!data.bottlenecks || data.bottlenecks.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No bottleneck data available</p></div>';
        return;
    }
    
    // Sort bottlenecks by impact score
    const bottlenecks = [...data.bottlenecks];
    bottlenecks.sort((a, b) => (b.impact_score || 0) - (a.impact_score || 0));
    
    // Create a treemap data
    const labels = bottlenecks.map(b => b.operation_name || 'Unknown');
    const parents = [''].concat(bottlenecks.slice(1).map(b => b.parent || ''));
    const values = bottlenecks.map(b => b.impact_score || 1);
    const colors = bottlenecks.map(b => getColorForImpact(b.impact_score || 0));
    
    const trace = {
        type: 'treemap',
        labels: labels,
        parents: parents,
        values: values,
        marker: { colors: colors },
        hovertemplate: '<b>%{label}</b><br>Impact: %{value:.2f}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Bottleneck Analysis',
        margin: {
            l: 0,
            r: 0,
            t: 50,
            b: 0
        },
        height: 500
    };
    
    // Create the plot
    Plotly.newPlot('bottleneck-visualization-container', [trace], layout);
    
    // Create recommendation cards for top bottlenecks
    const topBottlenecks = bottlenecks.slice(0, 5);
    const recommendationsHTML = `
        <div class="mt-4">
            <h5 class="mb-3">Top Bottleneck Recommendations</h5>
            <div class="row">
                ${topBottlenecks.map(b => `
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between">
                                <h6 class="m-0">${b.operation_name || 'Unknown'}</h6>
                                <span class="badge bg-danger">Impact: ${(b.impact_score || 0).toFixed(2)}</span>
                            </div>
                            <div class="card-body">
                                <p><strong>Category:</strong> ${b.category || 'N/A'}</p>
                                <p><strong>Recommendation:</strong> ${b.recommendation || 'No specific recommendation'}</p>
                                <p><strong>Potential Gain:</strong> ${b.potential_gain ? b.potential_gain.toFixed(1) + '%' : 'N/A'}</p>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Append recommendations
    container.innerHTML += recommendationsHTML;
    
    // Function to get color based on impact score
    function getColorForImpact(impact) {
        // Color scale from green (low impact) to red (high impact)
        if (impact < 0.2) return '#1cc88a';
        if (impact < 0.4) return '#36b9cc';
        if (impact < 0.6) return '#4e73df';
        if (impact < 0.8) return '#f6c23e';
        return '#e74a3b';
    }
}

/**
 * Create parameter sensitivity plot
 */
function createParameterSensitivityPlot(sensitivityData) {
    const container = document.getElementById('parameter-sensitivity-container');
    container.innerHTML = '';
    
    if (!sensitivityData || !sensitivityData.parameters || sensitivityData.parameters.length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No sensitivity data available</p></div>';
        return;
    }
    
    // Extract parameter data
    const parameters = sensitivityData.parameters;
    
    // Sort by sensitivity score
    parameters.sort((a, b) => (b.sensitivity || 0) - (a.sensitivity || 0));
    
    // Limit to top 10 parameters
    const topParameters = parameters.slice(0, 10);
    
    const paramNames = topParameters.map(p => p.name || 'Unknown');
    const sensitivity = topParameters.map(p => p.sensitivity || 0);
    
    // Create the trace
    const trace = {
        x: paramNames,
        y: sensitivity,
        type: 'bar',
        marker: {
            color: '#4e73df'
        },
        hovertemplate: '<b>%{x}</b><br>Sensitivity: %{y:.3f}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Parameter Sensitivity Analysis',
        xaxis: {
            title: 'Parameter',
            tickangle: -45
        },
        yaxis: {
            title: 'Sensitivity Score'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 100
        },
        hovermode: 'closest'
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Run comparison between selected results
 */
function runComparison() {
    const comparisonResults = document.getElementById('comparison-results');
    const selectedResults = Array.from(comparisonResults.options).map(option => option.value);
    
    if (selectedResults.length < 2) {
        alert('Please select at least two results to compare');
        return;
    }
    
    // Show the comparison results container
    document.getElementById('comparison-results-container').style.display = 'block';
    
    // Call the API to get comparison data
    fetch('/api/compare', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            result_ids: selectedResults
        })
    })
    .then(response => response.json())
    .then(data => {
        // Create comparison visualizations
        createComparativeTimeline(data);
        createSpeedupComparison(data);
        createMemoryComparison(data);
        createBreakdownComparison(data);
    })
    .catch(error => {
        console.error('Error running comparison:', error);
    });
}

/**
 * Create comparative timeline visualization
 */
function createComparativeTimeline(data) {
    const container = document.getElementById('comparative-timeline-container');
    container.innerHTML = '';
    
    if (!data.results || Object.keys(data.results).length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No comparison data available</p></div>';
        return;
    }
    
    // Extract result data
    const resultIds = Object.keys(data.results);
    const results = resultIds.map(id => data.results[id]);
    
    // Create a horizontal bar chart for each result
    const traces = [];
    
    resultIds.forEach((id, i) => {
        // Get total time for this result
        const result = results[i];
        const totalTime = result.total_time || result.avg_latency || 0;
        
        traces.push({
            x: [totalTime],
            y: [id],
            type: 'bar',
            orientation: 'h',
            name: id,
            marker: {
                color: i === 0 ? '#4e73df' : '#36b9cc'
            },
            hovertemplate: '<b>%{y}</b><br>Total time: %{x:.3f} ms<extra></extra>'
        });
    });
    
    // Create layout
    const layout = {
        title: 'Execution Time Comparison',
        xaxis: {
            title: 'Time (ms)'
        },
        yaxis: {
            title: 'Result ID'
        },
        margin: {
            l: 150,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest',
        barmode: 'group'
    };
    
    // Create the plot
    Plotly.newPlot(container, traces, layout);
}

/**
 * Create speedup comparison visualization
 */
function createSpeedupComparison(data) {
    const container = document.getElementById('speedup-comparison-container');
    container.innerHTML = '';
    
    if (!data.summary || Object.keys(data.summary).length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No speedup data available</p></div>';
        return;
    }
    
    // Extract summary data
    const resultIds = Object.keys(data.summary);
    const improvements = resultIds.map(id => ({
        id: id,
        latency: data.summary[id].latency_improvement || 0,
        throughput: data.summary[id].throughput_improvement || 0
    }));
    
    // Create grouped bar chart
    const latencyTrace = {
        x: improvements.map(imp => imp.id),
        y: improvements.map(imp => imp.latency),
        type: 'bar',
        name: 'Latency Improvement',
        marker: {
            color: '#4e73df'
        },
        text: improvements.map(imp => `${imp.latency.toFixed(1)}%`),
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Latency Improvement: %{text}<extra></extra>'
    };
    
    const throughputTrace = {
        x: improvements.map(imp => imp.id),
        y: improvements.map(imp => imp.throughput),
        type: 'bar',
        name: 'Throughput Improvement',
        marker: {
            color: '#1cc88a'
        },
        text: improvements.map(imp => `${imp.throughput.toFixed(1)}%`),
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Throughput Improvement: %{text}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Performance Improvements',
        xaxis: {
            title: 'Result ID'
        },
        yaxis: {
            title: 'Improvement (%)'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest',
        barmode: 'group'
    };
    
    // Create the plot
    Plotly.newPlot(container, [latencyTrace, throughputTrace], layout);
}

/**
 * Create memory comparison visualization
 */
function createMemoryComparison(data) {
    const container = document.getElementById('memory-comparison-container');
    container.innerHTML = '';
    
    if (!data.results || Object.keys(data.results).length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No memory data available</p></div>';
        return;
    }
    
    // Extract result data
    const resultIds = Object.keys(data.results);
    const results = resultIds.map(id => data.results[id]);
    
    // Extract peak memory for each result
    const peakMemory = results.map(result => (result.peak_memory || 0) / (1024 * 1024)); // Convert to MB
    
    // Create bar chart
    const trace = {
        x: resultIds,
        y: peakMemory,
        type: 'bar',
        marker: {
            color: '#36b9cc'
        },
        text: peakMemory.map(mem => `${mem.toFixed(2)} MB`),
        textposition: 'auto',
        hovertemplate: '<b>%{x}</b><br>Peak Memory: %{text}<extra></extra>'
    };
    
    // Create layout
    const layout = {
        title: 'Peak Memory Comparison',
        xaxis: {
            title: 'Result ID'
        },
        yaxis: {
            title: 'Memory (MB)'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest'
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
    
    // Add memory improvement annotations if we have summary data
    if (data.summary && Object.keys(data.summary).length > 0) {
        const summaryIds = Object.keys(data.summary);
        
        summaryIds.forEach(id => {
            const memoryImprovement = data.summary[id].memory_improvement || 0;
            
            // Add annotation for each improvement
            const index = resultIds.indexOf(id);
            if (index > 0) { // Skip first result (baseline)
                Plotly.relayout(container, {
                    annotations: [{
                        x: id,
                        y: peakMemory[index],
                        text: `${memoryImprovement.toFixed(1)}% reduction`,
                        showarrow: true,
                        arrowhead: 2,
                        ax: 0,
                        ay: -30
                    }]
                });
            }
        });
    }
}

/**
 * Create breakdown comparison visualization
 */
function createBreakdownComparison(data) {
    const container = document.getElementById('breakdown-comparison-container');
    container.innerHTML = '';
    
    if (!data.results || Object.keys(data.results).length === 0) {
        container.innerHTML = '<div class="text-center py-4"><p class="text-muted">No breakdown data available</p></div>';
        return;
    }
    
    // Extract result data
    const resultIds = Object.keys(data.results);
    const results = resultIds.map(id => data.results[id]);
    
    // Get all categories from all results
    const allCategories = new Set();
    results.forEach(result => {
        if (result.operations) {
            result.operations.forEach(op => {
                allCategories.add(op.category || 'Other');
            });
        }
    });
    const categories = Array.from(allCategories);
    
    // Calculate time spent in each category for each result
    const breakdowns = [];
    
    results.forEach(result => {
        const breakdown = {};
        categories.forEach(cat => { breakdown[cat] = 0; });
        
        if (result.operations) {
            result.operations.forEach(op => {
                const cat = op.category || 'Other';
                const duration = (op.end_time || 0) - (op.start_time || 0);
                breakdown[cat] += duration;
            });
        }
        
        breakdowns.push(breakdown);
    });
    
    // Create stacked bar chart
    const traces = [];
    
    categories.forEach((cat, i) => {
        traces.push({
            x: resultIds,
            y: breakdowns.map(b => b[cat]),
            type: 'bar',
            name: cat,
            marker: {
                color: getColorForCategory(cat, i)
            },
            hovertemplate: '<b>%{x}</b><br>' + cat + ': %{y:.3f} ms<extra></extra>'
        });
    });
    
    // Create layout
    const layout = {
        title: 'Operation Breakdown Comparison',
        xaxis: {
            title: 'Result ID'
        },
        yaxis: {
            title: 'Time (ms)'
        },
        margin: {
            l: 50,
            r: 50,
            t: 50,
            b: 50
        },
        hovermode: 'closest',
        barmode: 'stack',
        legend: {
            orientation: 'h',
            y: -0.2
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, traces, layout);
    
    // Function to get color for category
    function getColorForCategory(category, index) {
        const colors = [
            '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b',
            '#6f42c1', '#fd7e14', '#20c9a6', '#858796', '#5a5c69'
        ];
        
        return colors[index % colors.length];
    }
}

/**
 * Connect to live monitoring
 */
function connectToMonitoring() {
    const host = document.getElementById('monitoring-host').value || 'localhost';
    const port = document.getElementById('monitoring-port').value || '8085';
    const interval = document.getElementById('monitoring-interval').value || 1000;
    
    // Show the monitoring container
    document.getElementById('live-monitoring-container').style.display = 'block';
    
    // Here we would normally set up WebSocket connection
    // For this example, we'll simulate live data
    
    // Create initial plots
    createLiveLatencyPlot();
    createLiveThroughputPlot();
    createLiveMemoryPlot();
    createLiveGpuPlot();
    
    // Simulate data updates
    const monitoringInterval = setInterval(() => {
        updateLivePlots();
    }, parseInt(interval));
    
    // Store interval ID for cleanup
    window.monitoringInterval = monitoringInterval;
    
    // Update button to disconnect
    const connectButton = document.getElementById('connect-monitoring');
    connectButton.innerHTML = '<i class="fas fa-plug me-1"></i> Disconnect';
    connectButton.classList.remove('btn-primary');
    connectButton.classList.add('btn-danger');
    
    // Change button action to disconnect
    connectButton.onclick = function() {
        disconnectFromMonitoring();
    };
}

/**
 * Disconnect from live monitoring
 */
function disconnectFromMonitoring() {
    // Clear the update interval
    if (window.monitoringInterval) {
        clearInterval(window.monitoringInterval);
        window.monitoringInterval = null;
    }
    
    // Reset the UI
    document.getElementById('live-monitoring-container').style.display = 'none';
    
    // Reset button
    const connectButton = document.getElementById('connect-monitoring');
    connectButton.innerHTML = '<i class="fas fa-plug me-1"></i> Connect';
    connectButton.classList.remove('btn-danger');
    connectButton.classList.add('btn-primary');
    
    // Restore original connect function
    connectButton.onclick = function() {
        connectToMonitoring();
    };
}

/**
 * Create live latency plot
 */
function createLiveLatencyPlot() {
    const container = document.getElementById('live-latency-container');
    
    // Initial empty data
    const trace = {
        x: [],
        y: [],
        mode: 'lines',
        name: 'Latency',
        line: {
            color: '#4e73df',
            width: 2
        }
    };
    
    // Create layout
    const layout = {
        title: 'Live Latency',
        xaxis: {
            title: 'Time',
            range: [0, 30]
        },
        yaxis: {
            title: 'Latency (ms)'
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
    
    // Store initial timestamp
    window.monitoringStartTime = Date.now();
}

/**
 * Create live throughput plot
 */
function createLiveThroughputPlot() {
    const container = document.getElementById('live-throughput-container');
    
    // Initial empty data
    const trace = {
        x: [],
        y: [],
        mode: 'lines',
        name: 'Throughput',
        line: {
            color: '#1cc88a',
            width: 2
        }
    };
    
    // Create layout
    const layout = {
        title: 'Live Throughput',
        xaxis: {
            title: 'Time',
            range: [0, 30]
        },
        yaxis: {
            title: 'Throughput (samples/sec)'
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Create live memory plot
 */
function createLiveMemoryPlot() {
    const container = document.getElementById('live-memory-container');
    
    // Initial empty data
    const trace = {
        x: [],
        y: [],
        mode: 'lines',
        name: 'Memory',
        line: {
            color: '#36b9cc',
            width: 2
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(54, 185, 204, 0.1)'
    };
    
    // Create layout
    const layout = {
        title: 'Live Memory Usage',
        xaxis: {
            title: 'Time',
            range: [0, 30]
        },
        yaxis: {
            title: 'Memory (MB)'
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Create live GPU utilization plot
 */
function createLiveGpuPlot() {
    const container = document.getElementById('live-gpu-container');
    
    // Initial empty data
    const trace = {
        x: [],
        y: [],
        mode: 'lines',
        name: 'GPU',
        line: {
            color: '#f6c23e',
            width: 2
        }
    };
    
    // Create layout
    const layout = {
        title: 'Live GPU Utilization',
        xaxis: {
            title: 'Time',
            range: [0, 30]
        },
        yaxis: {
            title: 'Utilization (%)',
            range: [0, 100]
        },
        margin: {
            l: 50,
            r: 20,
            t: 50,
            b: 50
        }
    };
    
    // Create the plot
    Plotly.newPlot(container, [trace], layout);
}

/**
 * Update live monitoring plots with simulated data
 */
function updateLivePlots() {
    // Calculate elapsed time
    const elapsedTime = (Date.now() - window.monitoringStartTime) / 1000;
    
    // Generate simulated data
    const latency = 10 + Math.random() * 5 + Math.sin(elapsedTime / 10) * 3;
    const throughput = 100 + Math.random() * 20 + Math.cos(elapsedTime / 5) * 10;
    const memory = 500 + Math.random() * 50 + Math.sin(elapsedTime / 8) * 30;
    const gpu = 50 + Math.random() * 30 + Math.sin(elapsedTime / 3) * 20;
    
    // Update latency plot
    Plotly.extendTraces('live-latency-container', { x: [[elapsedTime]], y: [[latency]] }, [0]);
    
    // Adjust x-axis range to show last 30 seconds
    if (elapsedTime > 30) {
        Plotly.relayout('live-latency-container', {
            'xaxis.range': [elapsedTime - 30, elapsedTime]
        });
    }
    
    // Update throughput plot
    Plotly.extendTraces('live-throughput-container', { x: [[elapsedTime]], y: [[throughput]] }, [0]);
    
    if (elapsedTime > 30) {
        Plotly.relayout('live-throughput-container', {
            'xaxis.range': [elapsedTime - 30, elapsedTime]
        });
    }
    
    // Update memory plot
    Plotly.extendTraces('live-memory-container', { x: [[elapsedTime]], y: [[memory]] }, [0]);
    
    if (elapsedTime > 30) {
        Plotly.relayout('live-memory-container', {
            'xaxis.range': [elapsedTime - 30, elapsedTime]
        });
    }
    
    // Update GPU plot
    Plotly.extendTraces('live-gpu-container', { x: [[elapsedTime]], y: [[gpu]] }, [0]);
    
    if (elapsedTime > 30) {
        Plotly.relayout('live-gpu-container', {
            'xaxis.range': [elapsedTime - 30, elapsedTime]
        });
    }
}

/**
 * Move items between comparison lists
 */
function moveComparisonItems(sourceId, targetId) {
    const source = document.getElementById(sourceId);
    const target = document.getElementById(targetId);
    
    // Get selected options
    const selectedOptions = Array.from(source.selectedOptions);
    
    // Move selected options to target
    selectedOptions.forEach(option => {
        target.appendChild(option);
    });
}

/**
 * Upload result file
 */
function uploadResultFile() {
    const fileInput = document.getElementById('result-file');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select a file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    
    // Show progress
    const progressBar = document.querySelector('.progress');
    progressBar.style.display = 'block';
    
    // Hide previous messages
    document.getElementById('upload-success').style.display = 'none';
    document.getElementById('upload-error').style.display = 'none';
    
    // Send the file
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Hide progress
        progressBar.style.display = 'none';
        
        if (data.success) {
            // Show success message
            document.getElementById('upload-success').style.display = 'block';
            
            // Reload results list
            loadResults();
            
            // Reset form
            document.getElementById('upload-form').reset();
            
            // Close modal after a delay
            setTimeout(() => {
                const modal = bootstrap.Modal.getInstance(document.getElementById('uploadModal'));
                modal.hide();
            }, 1500);
        } else {
            // Show error message
            const errorElement = document.getElementById('upload-error');
            errorElement.textContent = data.error || 'Error uploading file';
            errorElement.style.display = 'block';
        }
    })
    .catch(error => {
        // Hide progress
        progressBar.style.display = 'none';
        
        // Show error message
        const errorElement = document.getElementById('upload-error');
        errorElement.textContent = 'Error: ' + error.message;
        errorElement.style.display = 'block';
    });
}

/**
 * Export current view as PNG
 */
function exportCurrentView() {
    // Get the currently visible view
    const activeView = document.querySelector('.view-section.active');
    
    if (!activeView) {
        return;
    }
    
    // Get all plots in the active view
    const plotContainers = activeView.querySelectorAll('.chart-container');
    
    if (plotContainers.length === 0) {
        alert('No visualizations to export in the current view');
        return;
    }
    
    // For simplicity, export the first plot
    const firstPlot = plotContainers[0];
    
    // Use Plotly's toImage function
    Plotly.toImage(firstPlot, {format: 'png', width: 800, height: 600})
        .then(function(dataUrl) {
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = dataUrl;
            downloadLink.download = 'ml-optimization-' + Date.now() + '.png';
            
            // Click the link
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        })
        .catch(function(error) {
            console.error('Error exporting image:', error);
            alert('Error exporting image');
        });
}

/**
 * Zoom timeline visualization
 */
function zoomTimeline(direction) {
    const container = document.getElementById('timeline-container');
    
    if (!container) {
        return;
    }
    
    // Get current x-axis range
    const currentLayout = container.layout || {};
    const currentXRange = currentLayout.xaxis ? currentLayout.xaxis.range : null;
    
    if (!currentXRange) {
        return;
    }
    
    // Calculate current range width
    const currentWidth = currentXRange[1] - currentXRange[0];
    
    // Calculate new range
    let newRange;
    
    if (direction === 'in') {
        // Zoom in: decrease range by 20%
        const newWidth = currentWidth * 0.8;
        const midpoint = (currentXRange[0] + currentXRange[1]) / 2;
        newRange = [midpoint - newWidth / 2, midpoint + newWidth / 2];
    } else {
        // Zoom out: increase range by 25%
        const newWidth = currentWidth * 1.25;
        const midpoint = (currentXRange[0] + currentXRange[1]) / 2;
        newRange = [midpoint - newWidth / 2, midpoint + newWidth / 2];
    }
    
    // Update layout
    Plotly.relayout(container, {
        'xaxis.range': newRange
    });
}

/**
 * Reset timeline view
 */
function resetTimelineView() {
    const container = document.getElementById('timeline-container');
    
    if (!container) {
        return;
    }
    
    // Reset to auto range
    Plotly.relayout(container, {
        'xaxis.autorange': true
    });
}

/**
 * Export timeline as PNG
 */
function exportTimelineAsPNG() {
    const container = document.getElementById('timeline-container');
    
    if (!container) {
        return;
    }
    
    // Use Plotly's toImage function
    Plotly.toImage(container, {format: 'png', width: 1200, height: 600})
        .then(function(dataUrl) {
            // Create download link
            const downloadLink = document.createElement('a');
            downloadLink.href = dataUrl;
            downloadLink.download = 'timeline-' + Date.now() + '.png';
            
            // Click the link
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        })
        .catch(function(error) {
            console.error('Error exporting timeline:', error);
            alert('Error exporting timeline');
        });
}

/**
 * Update charts for current theme
 */
function updateChartsForTheme() {
    const theme = document.documentElement.getAttribute('data-theme');
    const isDarkMode = theme === 'dark';
    
    // Get all plot containers
    const plotContainers = document.querySelectorAll('.chart-container');
    
    plotContainers.forEach(container => {
        // Skip empty containers
        if (!container.data) {
            return;
        }
        
        // Update layout colors for the current theme
        const layout = {
            paper_bgcolor: isDarkMode ? '#2a2e3f' : '#ffffff',
            plot_bgcolor: isDarkMode ? '#2a2e3f' : '#ffffff',
            font: {
                color: isDarkMode ? '#e9ecef' : '#5a5c69'
            },
            xaxis: {
                gridcolor: isDarkMode ? '#3a3f4b' : '#eaecef',
                zerolinecolor: isDarkMode ? '#3a3f4b' : '#eaecef'
            },
            yaxis: {
                gridcolor: isDarkMode ? '#3a3f4b' : '#eaecef',
                zerolinecolor: isDarkMode ? '#3a3f4b' : '#eaecef'
            }
        };
        
        // Apply the layout updates
        Plotly.relayout(container, layout);
    });
}