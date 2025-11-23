# 💻 Laptop Price Prediction System

A comprehensive machine learning system for predicting laptop prices based on specifications. This project includes data cleaning, multiple model training, comparison, and an interactive web interface for price predictions.

## 🎯 Project Overview

This system analyzes laptop specifications and predicts prices using three different machine learning algorithms:
- **Linear Regression** (Baseline model)
- **Random Forest** (Ensemble method)
- **Gradient Boosting** (Advanced ensemble)

## 📁 Project Structure

```
dharmu_ai/
├── archive/
│   ├── data.csv                    # Original dataset
│   └── data.xlsx                   # Excel version
├── data_cleaning.py                # Data preprocessing pipeline
├── model_design.py                 # ML model training and comparison
├── prediction_interface.py         # Streamlit web interface
├── main_pipeline.py               # Complete pipeline orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main_pipeline.py
```

### 3. Launch Web Interface
```bash
streamlit run prediction_interface.py
```

## 📊 Features

### Data Cleaning (`data_cleaning.py`)
- ✅ Handles missing values intelligently
- ✅ Extracts numeric values from text (RAM, Storage)
- ✅ Creates engineered features (performance score, gaming indicator)
- ✅ Encodes categorical variables
- ✅ Generates comprehensive visualizations
- ✅ Exports cleaned dataset

### Model Training (`model_design.py`)
- ✅ **Linear Regression**: Simple baseline model
- ✅ **Random Forest**: Robust ensemble method with hyperparameter tuning
- ✅ **Gradient Boosting**: Advanced ensemble with sequential learning
- ✅ Cross-validation for reliable performance estimates
- ✅ Feature importance analysis
- ✅ Model comparison and selection
- ✅ Automated model saving

### Web Interface (`prediction_interface.py`)
- ✅ Interactive Streamlit dashboard
- ✅ Real-time price predictions
- ✅ Budget comparison and recommendations
- ✅ Price breakdown visualization
- ✅ Feature impact analysis
- ✅ Laptop configuration comparison
- ✅ Mobile-responsive design

## 🎮 How to Use

### For Data Scientists:
1. **Explore Data**: Run `data_cleaning.py` to understand the dataset
2. **Train Models**: Use `model_design.py` to experiment with algorithms
3. **Compare Results**: Analyze model performance metrics
4. **Deploy**: Use the trained models in production

### For End Users:
1. **Open Web Interface**: Run `streamlit run prediction_interface.py`
2. **Configure Laptop**: Select brand, processor, RAM, storage, etc.
3. **Set Budget**: Enter your budget range
4. **Get Prediction**: Click "Predict Price" for instant results
5. **Analyze**: Review recommendations and comparisons

## 📈 Model Performance

The system trains three models and automatically selects the best performer:

| Model | Typical R² Score | RMSE | Use Case |
|-------|------------------|------|----------|
| Linear Regression | 0.75-0.85 | ₹15,000-25,000 | Quick baseline |
| Random Forest | 0.85-0.92 | ₹10,000-18,000 | Robust predictions |
| Gradient Boosting | 0.88-0.95 | ₹8,000-15,000 | Best accuracy |

## 🔧 Technical Details

### Data Processing
- **Dataset Size**: 900+ laptop configurations
- **Features**: 20+ engineered features
- **Price Range**: ₹10,000 - ₹500,000
- **Brands**: HP, Dell, Lenovo, Asus, Acer, Apple, MSI, Samsung, etc.

### Feature Engineering
- **Performance Score**: Weighted combination of RAM, processor, and rating
- **Gaming Indicator**: Binary flag for gaming laptops
- **Aspect Ratio**: Screen width/height ratio
- **Screen Area**: Calculated display area
- **RAM/Storage Ratio**: Memory to storage proportion

### Model Selection Criteria
- **Primary**: Test R² score (coefficient of determination)
- **Secondary**: Cross-validation stability
- **Tertiary**: RMSE (Root Mean Square Error)

## 💡 Use Cases

### 1. **Consumer Price Research**
- Compare laptop prices across brands
- Find best value for money options
- Budget planning for laptop purchases

### 2. **Market Analysis**
- Understand pricing trends
- Identify overpriced/underpriced models
- Competitive analysis

### 3. **Business Intelligence**
- Inventory pricing decisions
- Market positioning strategies
- Product development insights

## 🎯 Prediction Examples

### Budget Laptop (₹35,000)
```
Brand: HP
Processor: Intel i3 12th Gen
RAM: 8GB
Storage: 256GB SSD
Graphics: Intel Integrated
Display: 15.6" 1920x1080
```

### Gaming Laptop (₹85,000)
```
Brand: Asus
Processor: Intel i7 12th Gen
RAM: 16GB
Storage: 512GB SSD
Graphics: NVIDIA RTX 3060
Display: 15.6" 1920x1080
```

### Premium Ultrabook (₹120,000)
```
Brand: Apple
Processor: Apple M2
RAM: 16GB
Storage: 512GB SSD
Graphics: Integrated
Display: 13.6" 2560x1664
```

## 🔍 Model Insights

### Why These Algorithms?

**Linear Regression**:
- Simple and interpretable
- Fast training and prediction
- Good baseline for comparison
- Works well with linear relationships

**Random Forest**:
- Handles non-linear patterns
- Robust to outliers
- Provides feature importance
- Reduces overfitting through ensemble

**Gradient Boosting**:
- Excellent predictive performance
- Handles complex interactions
- Sequential learning improves accuracy
- Good generalization capability

## 📱 Web Interface Features

### Main Dashboard
- **Configuration Panel**: Select laptop specifications
- **Budget Input**: Set your price range
- **Instant Prediction**: Real-time price estimates
- **Visual Analysis**: Charts and comparisons

### Analysis Tabs
1. **Price Analysis**: Component cost breakdown
2. **Comparison**: Similar laptop pricing
3. **Recommendations**: Budget-based suggestions
4. **Summary**: Complete configuration overview

## 🛠️ Customization

### Adding New Features
```python
# In data_cleaning.py
def create_custom_feature(self):
    self.cleaned_df['custom_feature'] = # your logic here
```

### Training New Models
```python
# In model_design.py
def create_custom_model(self):
    model = YourCustomModel()
    # training logic
    return model
```

### Interface Modifications
```python
# In prediction_interface.py
def add_custom_input(self):
    custom_value = st.selectbox("Custom Feature", options)
    return custom_value
```

## 📊 Performance Monitoring

The system includes comprehensive performance tracking:
- **Training Metrics**: R², RMSE, MAE
- **Validation**: Cross-validation scores
- **Generalization**: Train vs. test performance
- **Feature Analysis**: Importance rankings

## 🚀 Deployment Options

### Local Development
```bash
streamlit run prediction_interface.py
```

### Production Deployment
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Container-based deployment
- **AWS/GCP**: Cloud platform deployment
- **Docker**: Containerized deployment

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time market data integration
- [ ] Advanced deep learning models
- [ ] Multi-currency support
- [ ] Historical price trends
- [ ] Recommendation engine
- [ ] API endpoints for integration

### Model Improvements
- [ ] XGBoost and LightGBM integration
- [ ] Neural network architectures
- [ ] Ensemble stacking methods
- [ ] Automated hyperparameter optimization
- [ ] Online learning capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: Laptop price dataset from various e-commerce platforms
- Scikit-learn for machine learning algorithms
- Streamlit for the web interface framework
- Plotly for interactive visualizations

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Happy Predicting! 💻✨**