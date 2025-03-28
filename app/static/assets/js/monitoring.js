// 初始化ECharts实例
function initLatencyChart() {
    const chart = echarts.init(document.getElementById('latency-chart'));
    
    const option = {
        darkMode: true,
        tooltip: { trigger: 'axis' },
        xAxis: {
            type: 'category',
            data: [],
            axisLabel: { color: '#adb5bd' }
        },
        yAxis: {
            type: 'value',
            axisLabel: { color: '#adb5bd' },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
        },
        series: [{
            name: '请求延迟',
            type: 'line',
            smooth: true,
            data: [],
            itemStyle: { color: '#20c997' },
            areaStyle: { color: 'rgba(32,201,151,0.3)' }
        }]
    };
    chart.setOption(option);
    return chart;
}

// 建立SSE连接
const eventSource = new EventSource('/monitoring/stream');

// 初始化图表
const latencyChart = initLatencyChart();

// 实时更新处理
eventSource.addEventListener('metrics', (e) => {
    const data = JSON.parse(e.data);
    
    // 更新RPS计数器
    document.getElementById('rps-counter').innerText = data.rps.toFixed(1);
    
    // 更新延迟图表
    const option = latencyChart.getOption();
    const timestamp = new Date().toLocaleTimeString();
    
    if (option.xAxis[0].data.length > 15) {
        option.xAxis[0].data.shift();
        option.series[0].data.shift();
    }
    
    option.xAxis[0].data.push(timestamp);
    option.series[0].data.push(data.latency.avg);
    
    latencyChart.setOption(option);
});

// 窗口关闭时断开连接
window.addEventListener('beforeunload', () => {
    eventSource.close();
});