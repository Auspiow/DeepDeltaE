fetch('./color_comparison_results.json')
  .then(response => response.json())
  .then(data => {
    // --- 1. 计算差值并排序 ---
    const sortedData = data.map(item => {
        const humanScore = parseFloat(item.human_score);
        const modelScore = parseFloat(item.model_score);
        const e2000Score = parseFloat(item.e2000_score); // 获取 ΔE2000 分数

        // 您的模型误差 (用于排序)
        const modelDifference = Math.abs(humanScore - modelScore); 
        // ΔE2000 模型误差
        const e2000Difference = Math.abs(humanScore - e2000Score); 

        return {
            ...item,
            difference: modelDifference, // 您的模型误差
            e2000Difference: e2000Difference // ΔE2000 误差
        };
    })
    .sort((a, b) => a.difference - b.difference); // 列表按您的模型误差升序排序

    // --- 2. 渲染已排序的数据 ---
    const container = document.getElementById('color-comparisons');
    container.innerHTML = ''; // 清空现有内容

    const formatScore = (score) => parseFloat(score).toFixed(2);

    sortedData.forEach(item => {
      const colorComparison = document.createElement('div');
      colorComparison.classList.add('color-comparison');
      
      const colorBoxContainer = document.createElement('div');
      colorBoxContainer.classList.add('color-box-container');

      // 创建颜色块 1
      const colorBox1 = document.createElement('div');
      colorBox1.classList.add('color-box');
      colorBox1.style.backgroundColor = `rgb(${item.color1.r}, ${item.color1.g}, ${item.color1.b})`;

      // 创建颜色块 2
      const colorBox2 = document.createElement('div');
      colorBox2.classList.add('color-box');
      colorBox2.style.backgroundColor = `rgb(${item.color2.r}, ${item.color2.g}, ${item.color2.b})`;

      colorBoxContainer.appendChild(colorBox1);
      colorBoxContainer.appendChild(colorBox2);
      
      // 创建评分文本
      const scores = document.createElement('div');
      scores.classList.add('scores');

      const modelDiffScore = formatScore(item.difference);
      const e2000DiffScore = formatScore(item.e2000Difference);

      // 确定哪个模型的误差更小（即性能更好）
      const isModelBetter = item.difference < item.e2000Difference;
      const modelClass = isModelBetter ? 'highlight-win' : 'highlight-loss';
      const e2000Class = !isModelBetter ? 'highlight-win' : 'highlight-loss';


      scores.innerHTML = `
        <p><strong>Human Score (ΔE raw):</strong> ${formatScore(item.human_score)}</p>
        <p><strong>ΔE2000 Score:</strong> ${formatScore(item.e2000_score)}</p>
        <p><strong>Model Prediction (ΔE_vis):</strong> ${formatScore(item.model_score)}</p>
        
        <hr style="border: 0; border-top: 1px solid #f0f0f0; margin: 8px 0;">
        
        <p class="difference-score">
            <strong>模型预测误差对比 (|ΔE raw - ΔE pred|):</strong>
        </p>
        
        <div class="error-comparison-row">
            <span class="error-label">ΔE2000 误差:</span>
            <span class="error-value ${e2000Class}">${e2000DiffScore}</span>
        </div>
        <div class="error-comparison-row">
            <span class="error-label">Siamese模型误差:</span>
            <span class="error-value ${modelClass}">${modelDiffScore}</span>
        </div>
      `;

      // 将颜色块容器和评分文本添加到比较容器中
      colorComparison.appendChild(colorBoxContainer);
      colorComparison.appendChild(scores);

      container.appendChild(colorComparison);
    });
  })
  .catch(error => console.error('Error loading data:', error));