// ===================================================================
// Supply Chain Environment Dashboard - Chart Configuration
// Uses Chart.js (loaded from CDN)
// ===================================================================

const CHART_CONFIG = {
  colors: {
    primary: '#00d4ff',
    secondary: '#0066cc',
    success: '#51cf66',
    warning: '#ff9500',
    danger: '#ff6b6b',
    text: '#e0e0e0',
    textSecondary: '#a0a0a0',
    border: '#2a3f5f',
  },
  chartOptions: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index',
    },
    plugins: {
      legend: {
        display: true,
        labels: {
          color: '#e0e0e0',
          font: { size: 11 },
          usePointStyle: true,
          padding: 15,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(26, 38, 51, 0.9)',
        titleColor: '#00d4ff',
        bodyColor: '#e0e0e0',
        borderColor: '#2a3f5f',
        borderWidth: 1,
        padding: 10,
        displayColors: true,
        callbacks: {
          labelColor: function(context) {
            return {
              borderColor: context.borderColor,
              backgroundColor: context.borderColor,
            };
          },
        },
      },
    },
  },
};

class DashboardCharts {
  constructor() {
    this.charts = {};
    this.maxDataPoints = 30;
    this.serviceLevelData = [];
    this.costData = [];
    this.disruptionData = [];
    this.rewardData = [];
  }

  initCharts() {
    this.initServiceLevelChart();
    this.initCostChart();
    this.initDisruptionChart();
    this.initRewardChart();
  }

  initServiceLevelChart() {
    const ctx = document.getElementById('serviceLevelChart')?.getContext('2d');
    if (!ctx) return;

    this.charts.serviceLevel = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Service Level',
            data: [],
            borderColor: CHART_CONFIG.colors.primary,
            backgroundColor: 'rgba(0, 212, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            pointRadius: 3,
            pointBorderColor: CHART_CONFIG.colors.primary,
            pointBackgroundColor: CHART_CONFIG.colors.primary,
            tension: 0.4,
          },
          {
            label: 'Target',
            data: [],
            borderColor: CHART_CONFIG.colors.warning,
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            tension: 0,
          },
        ],
      },
      options: {
        ...CHART_CONFIG.chartOptions,
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { color: CHART_CONFIG.colors.textSecondary, stepSize: 0.2 },
            grid: { color: CHART_CONFIG.colors.border },
          },
          x: {
            ticks: { color: CHART_CONFIG.colors.textSecondary },
            grid: { color: CHART_CONFIG.colors.border },
          },
        },
      },
    });
  }

  initCostChart() {
    const ctx = document.getElementById('costChart')?.getContext('2d');
    if (!ctx) return;

    this.charts.cost = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Total Cost',
            data: [],
            borderColor: CHART_CONFIG.colors.danger,
            backgroundColor: 'rgba(255, 107, 107, 0.1)',
            borderWidth: 2,
            fill: true,
            pointRadius: 3,
            pointBorderColor: CHART_CONFIG.colors.danger,
            pointBackgroundColor: CHART_CONFIG.colors.danger,
            tension: 0.4,
          },
        ],
      },
      options: {
        ...CHART_CONFIG.chartOptions,
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: { color: CHART_CONFIG.colors.textSecondary, stepSize: 0.2 },
            grid: { color: CHART_CONFIG.colors.border },
          },
          x: {
            ticks: { color: CHART_CONFIG.colors.textSecondary },
            grid: { color: CHART_CONFIG.colors.border },
          },
        },
      },
    });
  }

  initDisruptionChart() {
    const ctx = document.getElementById('disruptionChart')?.getContext('2d');
    if (!ctx) return;

    this.charts.disruption = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Active Disruptions',
            data: [],
            backgroundColor: CHART_CONFIG.colors.warning,
            borderColor: CHART_CONFIG.colors.warning,
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...CHART_CONFIG.chartOptions,
        scales: {
          y: {
            min: 0,
            ticks: { color: CHART_CONFIG.colors.textSecondary, stepSize: 1 },
            grid: { color: CHART_CONFIG.colors.border },
          },
          x: {
            ticks: { color: CHART_CONFIG.colors.textSecondary },
            grid: { color: CHART_CONFIG.colors.border },
          },
        },
      },
    });
  }

  initRewardChart() {
    const ctx = document.getElementById('rewardChart')?.getContext('2d');
    if (!ctx) return;

    this.charts.reward = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Cumulative Reward',
            data: [],
            borderColor: CHART_CONFIG.colors.success,
            backgroundColor: 'rgba(81, 207, 102, 0.1)',
            borderWidth: 2,
            fill: true,
            pointRadius: 3,
            pointBorderColor: CHART_CONFIG.colors.success,
            pointBackgroundColor: CHART_CONFIG.colors.success,
            tension: 0.4,
          },
        ],
      },
      options: {
        ...CHART_CONFIG.chartOptions,
        scales: {
          y: {
            ticks: { color: CHART_CONFIG.colors.textSecondary },
            grid: { color: CHART_CONFIG.colors.border },
          },
          x: {
            ticks: { color: CHART_CONFIG.colors.textSecondary },
            grid: { color: CHART_CONFIG.colors.border },
          },
        },
      },
    });
  }

  addDataPoint(stepNum, serviceLevel, cost, disruptionCount, cumulativeReward, target = 0.75) {
    const label = `Step ${stepNum}`;

    // Service Level
    if (this.charts.serviceLevel) {
      this.charts.serviceLevel.data.labels.push(label);
      this.charts.serviceLevel.data.datasets[0].data.push(Math.round(serviceLevel * 100) / 100);
      this.charts.serviceLevel.data.datasets[1].data.push(Math.round(target * 100) / 100);
      this.trimData(this.charts.serviceLevel);
      this.charts.serviceLevel.update('none');
    }

    // Cost
    if (this.charts.cost) {
      this.charts.cost.data.labels.push(label);
      this.charts.cost.data.datasets[0].data.push(Math.round(cost * 100) / 100);
      this.trimData(this.charts.cost);
      this.charts.cost.update('none');
    }

    // Disruption
    if (this.charts.disruption) {
      this.charts.disruption.data.labels.push(label);
      this.charts.disruption.data.datasets[0].data.push(disruptionCount);
      this.trimData(this.charts.disruption);
      this.charts.disruption.update('none');
    }

    // Reward
    if (this.charts.reward) {
      this.charts.reward.data.labels.push(label);
      this.charts.reward.data.datasets[0].data.push(Math.round(cumulativeReward * 100) / 100);
      this.trimData(this.charts.reward);
      this.charts.reward.update('none');
    }
  }

  trimData(chart) {
    const datasets = chart.data.datasets;
    if (chart.data.labels.length > this.maxDataPoints) {
      chart.data.labels.shift();
      datasets.forEach(ds => ds.data.shift());
    }
  }

  clearAll() {
    Object.values(this.charts).forEach(chart => {
      if (chart) {
        chart.data.labels = [];
        chart.data.datasets.forEach(ds => ds.data = []);
        chart.update('none');
      }
    });
    this.serviceLevelData = [];
    this.costData = [];
    this.disruptionData = [];
    this.rewardData = [];
  }

  destroy() {
    Object.values(this.charts).forEach(chart => {
      if (chart) chart.destroy();
    });
    this.charts = {};
  }
}

// Export for use in dashboard
const dashboardCharts = new DashboardCharts();
