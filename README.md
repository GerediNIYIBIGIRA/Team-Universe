# RwandaYouthWorks: Tackling Youth Unemployment in Rwanda

## Project Overview

The RwandaYouthWorks project addresses Rwanda's critical youth unemployment crisis. With a youth unemployment rate of 20.50% in 2024 significantly higher than the overall unemployment rate of 16.80% this issue threatens the country's social stability and economic development. Despite Rwanda’s remarkable GDP growth averaging 7.5% since 2000, the economy has struggled to generate sufficient quality jobs for youth, who make up over 70% of the population. Challenges such as gender disparities, skills mismatches, and widespread underemployment further exacerbate the problem, with 60% of employed youth working in nonproductive sectors.

This project leverages data science and machine learning to deliver actionable insights, evidence-based policy recommendations, and targeted career guidance to mitigate youth unemployment in Rwanda.

### Objectives
1. Analyze Youth Unemployment Trends:
  . Investigate demographic and regional patterns to identify disparities.
  . Assess the impact of education, gender, and sectoral employment on youth job opportunities.

2. Forecast Employment Patterns:
   . Use predictive models to estimate unemployment rates for 2025–2030.
3. Support Policy Development:
   . Provide data-driven insights to guide interventions addressing skills mismatches and underemployment.
4. Empower Youth:
   . Develop tools for career guidance and resource access tailored to young job seekers.
   
### Data Source
Primary Dataset: Labour Force Survey 2019 from the National Institute of Statistics of Rwanda (NISR).

### Relevance to Objectives
 . Demographics: Youth-focused age groups (age5, age10), gender.
 . Education: Educational attainment (attained), vocational training (E01B1).
 . Employment Metrics: Employment status (B01), sectoral employment (B08), and NEET indicators (neetyoung).
 . Regional Insights: Province and district-level data (province, code_dis).
 . Unemployment Indicators: Unemployment rate (UR1), labor force participation rate (LFPR).
 
### Solution Features
1. Interactive Dashboard:
   . Displays unemployment trends by age, gender, education, and location.
   . Visualizes sector-specific and regional disparities in youth employment.
   
2. Unemployment Forecasting:
Predicts youth unemployment trends for 2025–2030 using ARIMA and machine learning models.

3. AI-Powered Chatbot:
  . Provides career guidance and personalized resources for job seekers.
  . Supports policymakers with quick access to insights and trends.
   
4. Data-Driven Recommendations:
  . Suggests actionable interventions to address underemployment, skills mismatches, and gender disparities.
   
### Technical Details

. Programming Languages: Python.
. Key Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Dash.
. Modeling:
  . Target variable: target_employed16 (binary classification for employment prediction).
  . Models: ARIMA for forecasting, classification models for trend analysis.
. Deployment:
  . Interactive dashboard hosted using Dash.
  . Flask backend for API and chatbot integration.
  
### Impact
The RwandaYouthWorks project is designed to:

. Help policymakers address regional and sectoral disparities.
. Improve vocational training programs to align with labor market demands.
. Equip youth with actionable career guidance.
. Drive Rwanda's social and economic development by connecting growth with quality youth employment.
