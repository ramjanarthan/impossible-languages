// Main application logic
class ExperimentsApp {
    constructor() {
        this.experiments = {};
        this.tooltip = document.getElementById('tooltip');
        this.init();
    }

    async init() {
        try {
            await this.loadExperiments();
            this.renderExperiments();
            this.setupEventListeners();
        } catch (error) {
            console.error('Error initializing app:', error);
            this.showError('Failed to load experiments. Please check the console for details.');
        }
    }

    async loadExperiments() {
        try {
            const response = await fetch('/api/experiments');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.experiments = await response.json();
        } catch (error) {
            console.error('Error loading experiments:', error);
            throw error;
        }
    }

    renderExperiments() {
        const loading = document.getElementById('loading');
        const container = document.getElementById('experiments-container');
        const noData = document.getElementById('no-data');

        loading.style.display = 'none';

        if (Object.keys(this.experiments).length === 0) {
            noData.style.display = 'block';
            return;
        }

        container.style.display = 'block';
        container.innerHTML = '';

        // Sort phenomena alphabetically
        const sortedPhenomena = Object.keys(this.experiments).sort();

        sortedPhenomena.forEach(phenomenon => {
            const phenomenonSection = this.createPhenomenonSection(phenomenon, this.experiments[phenomenon]);
            container.appendChild(phenomenonSection);
        });
    }

    createPhenomenonSection(phenomenon, experiments) {
        const section = document.createElement('div');
        section.className = 'phenomenon-section';

        const title = document.createElement('h2');
        title.className = 'phenomenon-title';
        title.textContent = this.formatPhenomenonName(phenomenon);

        const grid = document.createElement('div');
        grid.className = 'experiments-grid';

        experiments.forEach(experiment => {
            const card = this.createExperimentCard(experiment);
            grid.appendChild(card);
        });

        section.appendChild(title);
        section.appendChild(grid);

        return section;
    }

    createExperimentCard(experiment) {
        const card = document.createElement('div');
        card.className = 'experiment-card';

        const header = document.createElement('div');
        header.className = 'experiment-header';

        const languages = document.createElement('div');
        languages.className = 'experiment-languages';
        languages.textContent = `${this.formatLanguageName(experiment.language1)} → ${this.formatLanguageName(experiment.language2)}`;

        const models = document.createElement('div');
        models.className = 'experiment-models';
        models.innerHTML = `
            <div><strong>Model 1:</strong> ${this.formatModelName(experiment.model1)}</div>
            <div><strong>Model 2:</strong> ${this.formatModelName(experiment.model2)}</div>
        `;

        header.appendChild(languages);
        header.appendChild(models);

        const comparison = this.createModelsComparison(experiment);
        const info = this.createExperimentInfo(experiment);

        card.appendChild(header);
        card.appendChild(comparison);
        card.appendChild(info);

        return card;
    }

    createModelsComparison(experiment) {
        const comparison = document.createElement('div');
        comparison.className = 'models-comparison';

        const model1Box = this.createModelBox(
            experiment.model1,
            experiment.accuracy1,
            experiment.accuracy1 >= experiment.accuracy2,
            experiment,
            'model1'
        );

        const model2Box = this.createModelBox(
            experiment.model2,
            experiment.accuracy2,
            experiment.accuracy2 > experiment.accuracy1,
            experiment,
            'model2'
        );

        comparison.appendChild(model1Box);
        comparison.appendChild(model2Box);

        return comparison;
    }

    createModelBox(modelName, accuracy, isWinner, experiment, modelType) {
        const box = document.createElement('div');
        box.className = `model-box ${isWinner ? 'winner' : 'loser'}`;

        const name = document.createElement('div');
        name.className = 'model-name';
        name.textContent = this.formatModelName(modelName);

        const accuracyDiv = document.createElement('div');
        accuracyDiv.className = 'model-accuracy';
        accuracyDiv.textContent = `${accuracy.toFixed(1)}%`;

        box.appendChild(name);
        box.appendChild(accuracyDiv);

        // Add tooltip data
        box.setAttribute('data-experiment', JSON.stringify(experiment));
        box.setAttribute('data-model-type', modelType);

        return box;
    }

    createExperimentInfo(experiment) {
        const info = document.createElement('div');
        info.className = 'experiment-info';

        const totalPairs = document.createElement('div');
        totalPairs.className = 'info-item';
        totalPairs.innerHTML = `
            <span class="info-label">Total Pairs:</span>
            <span>${experiment.total_pairs}</span>
        `;

        const timestamp = document.createElement('div');
        timestamp.className = 'info-item';
        timestamp.innerHTML = `
            <span class="info-label">Timestamp:</span>
            <span>${this.formatTimestamp(experiment.timestamp)}</span>
        `;

        info.appendChild(totalPairs);
        info.appendChild(timestamp);

        return info;
    }

    setupEventListeners() {
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.model-box')) {
                this.showTooltip(e);
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('.model-box')) {
                this.hideTooltip();
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (e.target.closest('.model-box')) {
                this.updateTooltipPosition(e);
            }
        });
    }

    showTooltip(event) {
        const modelBox = event.target.closest('.model-box');
        const experiment = JSON.parse(modelBox.getAttribute('data-experiment'));
        const modelType = modelBox.getAttribute('data-model-type');

        const tooltipData = document.getElementById('tooltip-data');
        tooltipData.innerHTML = this.generateTooltipContent(experiment, modelType);

        this.tooltip.classList.add('show');
        this.updateTooltipPosition(event);
    }

    hideTooltip() {
        this.tooltip.classList.remove('show');
    }

    updateTooltipPosition(event) {
        const tooltipRect = this.tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let left = event.pageX + 15;
        let top = event.pageY - 10;

        // Adjust position to keep tooltip in viewport
        if (left + tooltipRect.width > viewportWidth) {
            left = event.pageX - tooltipRect.width - 15;
        }

        if (top + tooltipRect.height > viewportHeight) {
            top = event.pageY - tooltipRect.height - 10;
        }

        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
    }

    generateTooltipContent(experiment, modelType) {
        let content = '<div class="tooltip-metrics">';

        // Add perplexity metrics
        if (experiment.perplexities && Object.keys(experiment.perplexities).length > 0) {
            content += '<div class="tooltip-section"><strong>Perplexity Metrics:</strong>';
            Object.entries(experiment.perplexities).forEach(([key, value]) => {
                const formattedKey = this.formatPerplexityKey(key);
                content += `
                    <div class="tooltip-metric">
                        <span class="metric-label">${formattedKey}:</span>
                        <span class="metric-value">${value.toFixed(2)}</span>
                    </div>
                `;
            });
            content += '</div>';
        }

        // Add comparison counts
        if (experiment.comparison_counts && Object.keys(experiment.comparison_counts).length > 0) {
            content += '<div class="tooltip-section"><strong>Comparison Counts:</strong>';
            Object.entries(experiment.comparison_counts).forEach(([key, value]) => {
                const formattedKey = this.formatComparisonKey(key);
                content += `
                    <div class="tooltip-metric">
                        <span class="metric-label">${formattedKey}:</span>
                        <span class="metric-value">${value}</span>
                    </div>
                `;
            });
            content += '</div>';
        }

        content += '</div>';
        return content;
    }

    // Utility methods for formatting
    formatPhenomenonName(phenomenon) {
        return phenomenon.replace(/_/g, ' ').toLowerCase();
    }

    formatLanguageName(language) {
        return language.replace(/_/g, ' ').split(' ').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    formatModelName(modelName) {
        // Extract the last part after the last slash
        const parts = modelName.split('/');
        return parts[parts.length - 1].replace(/-/g, ' ').replace(/_/g, ' ');
    }

    formatTimestamp(timestamp) {
        const year = timestamp.substring(0, 4);
        const month = timestamp.substring(4, 6);
        const day = timestamp.substring(6, 8);
        const hour = timestamp.substring(9, 11);
        const minute = timestamp.substring(11, 13);
        const second = timestamp.substring(13, 15);

        return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
    }

    formatPerplexityKey(key) {
        // Convert keys like "A_good_m1" to "A Good M1"
        return key.replace(/_/g, ' ').toUpperCase();
    }

    formatComparisonKey(key) {
        // Convert keys like "both_correct" to "Both Correct"
        return key.replace(/_/g, ' ').split(' ').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    showError(message) {
        const loading = document.getElementById('loading');
        const container = document.getElementById('experiments-container');
        const noData = document.getElementById('no-data');

        loading.style.display = 'none';
        container.style.display = 'none';
        noData.style.display = 'block';

        const noDataMessage = noData.querySelector('.no-data-message');
        if (noDataMessage) {
            noDataMessage.innerHTML = `
                <h2>Error Loading Experiments</h2>
                <p>${message}</p>
            `;
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ExperimentsApp();
});