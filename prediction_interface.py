import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class LaptopPricePredictionInterface:
    def __init__(self):
        """Initialize the prediction interface"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.brand_encoder = None
        self.processor_encoder = None
        self.gpu_encoder = None
        self.os_encoder = None
        
    def load_model_and_encoders(self, model_path="models/best_model.pkl", scaler_path="models/scaler.pkl"):
        """Load trained model and preprocessing components"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            st.success("✅ Model and preprocessing components loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def create_sidebar_inputs(self):
        """Create sidebar inputs for laptop specifications"""
        st.sidebar.header("🔧 Laptop Specifications")
        
        # Brand selection
        brands = ['HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'Apple', 'MSI', 'Samsung', 'Infinix', 'Xiaomi']
        brand = st.sidebar.selectbox("Brand", brands)
        
        # Processor selection
        processors = [
            '12th Gen Intel Core i5', '13th Gen Intel Core i7', '11th Gen Intel Core i5',
            '5th Gen AMD Ryzen 5', '7th Gen AMD Ryzen 7', '12th Gen Intel Core i7',
            '13th Gen Intel Core i5', '11th Gen Intel Core i7', '5th Gen AMD Ryzen 7'
        ]
        processor = st.sidebar.selectbox("Processor", processors)
        
        # RAM
        ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
        
        # Storage
        storage = st.sidebar.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048])
        storage_type = st.sidebar.selectbox("Storage Type", ['SSD', 'Hard-Disk'])
        
        # Display
        display_size = st.sidebar.slider("Display Size (inches)", 11.0, 18.0, 15.6, 0.1)
        resolution = st.sidebar.selectbox("Resolution", 
                                        ['1366x768', '1920x1080', '2560x1440', '3840x2160'])
        
        # GPU
        gpu_options = [
            'Intel Integrated', 'Intel Iris Xe', 'NVIDIA GeForce RTX 3050',
            'NVIDIA GeForce RTX 4060', 'AMD Radeon', 'NVIDIA GeForce GTX 1650'
        ]
        gpu = st.sidebar.selectbox("Graphics Card", gpu_options)
        
        # Operating System
        os_options = ['Windows 11 OS', 'Windows 10 OS', 'Mac OS', 'DOS OS', 'Chrome OS']
        operating_system = st.sidebar.selectbox("Operating System", os_options)
        
        # Additional features
        st.sidebar.subheader("📊 Additional Features")
        spec_rating = st.sidebar.slider("Specification Rating", 50.0, 90.0, 70.0, 0.1)
        warranty = st.sidebar.selectbox("Warranty (Years)", [0, 1, 2, 3])
        
        return {
            'brand': brand,
            'processor': processor,
            'ram': ram,
            'storage': storage,
            'storage_type': storage_type,
            'display_size': display_size,
            'resolution': resolution,
            'gpu': gpu,
            'os': operating_system,
            'spec_rating': spec_rating,
            'warranty': warranty
        }
    
    def preprocess_inputs(self, inputs):
        """Preprocess user inputs to match model requirements"""
        # Parse resolution
        width, height = map(int, inputs['resolution'].split('x'))
        
        # Create feature vector (simplified version - in real implementation, 
        # you'd need the exact same preprocessing as training)
        features = {
            'spec_rating': inputs['spec_rating'],
            'Ram_numeric': inputs['ram'],
            'ROM_numeric': inputs['storage'],
            'display_size_numeric': inputs['display_size'],
            'resolution_width': width,
            'resolution_height': height,
            'warranty': inputs['warranty'],
            'aspect_ratio': width / height,
            'screen_area': (inputs['display_size'] ** 2) * 0.5,
            'ram_rom_ratio': inputs['ram'] / inputs['storage'],
            'performance_score': inputs['ram'] * 0.3 + 70 * 0.4 + inputs['spec_rating'] * 0.3,
            'is_gaming': 1 if 'RTX' in inputs['gpu'] or 'GTX' in inputs['gpu'] else 0,
            'processor_gen': 12  # Default value
        }
        
        # Add encoded categorical features (simplified)
        brand_mapping = {'HP': 0, 'Dell': 1, 'Lenovo': 2, 'Asus': 3, 'Acer': 4, 'Apple': 5, 'MSI': 6, 'Samsung': 7, 'Infinix': 8, 'Xiaomi': 9}
        features['brand_encoded'] = brand_mapping.get(inputs['brand'], 0)
        
        processor_mapping = {'12th Gen Intel Core i5': 0, '13th Gen Intel Core i7': 1, '11th Gen Intel Core i5': 2}
        features['processor_encoded'] = processor_mapping.get(inputs['processor'], 0)
        
        gpu_mapping = {'Intel Integrated': 0, 'Intel Iris Xe': 1, 'NVIDIA GeForce RTX 3050': 2}
        features['GPU_encoded'] = gpu_mapping.get(inputs['gpu'], 0)
        
        os_mapping = {'Windows 11 OS': 0, 'Windows 10 OS': 1, 'Mac OS': 2, 'DOS OS': 3, 'Chrome OS': 4}
        features['OS_encoded'] = os_mapping.get(inputs['os'], 0)
        
        # Add remaining encoded features with default values
        features.update({
            'CPU_encoded': 0,
            'Ram_type_encoded': 0,
            'ROM_type_encoded': 0 if inputs['storage_type'] == 'SSD' else 1
        })
        
        return features
    
    def predict_price(self, features):
        """Make price prediction"""
        try:
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Ensure all required features are present (add missing ones with default values)
            required_features = [
                'brand_encoded', 'spec_rating', 'processor_encoded', 'CPU_encoded',
                'Ram_numeric', 'Ram_type_encoded', 'ROM_numeric', 'ROM_type_encoded',
                'GPU_encoded', 'display_size_numeric', 'resolution_width', 'resolution_height',
                'OS_encoded', 'warranty', 'aspect_ratio', 'screen_area', 'ram_rom_ratio',
                'performance_score', 'is_gaming', 'processor_gen'
            ]
            
            for feature in required_features:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            # Reorder columns to match training data
            feature_df = feature_df[required_features]
            
            # Make prediction
            if hasattr(self.model, 'predict'):
                # For tree-based models (Random Forest, Gradient Boosting)
                prediction = self.model.predict(feature_df)[0]
            else:
                # For linear models (need scaling)
                scaled_features = self.scaler.transform(feature_df)
                prediction = self.model.predict(scaled_features)[0]
            
            return max(0, prediction)  # Ensure non-negative price
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def create_price_breakdown(self, inputs, predicted_price):
        """Create price breakdown visualization"""
        # Estimate component contributions (simplified)
        base_price = 20000
        
        # Brand premium
        brand_premium = {'Apple': 30000, 'MSI': 15000, 'Asus': 10000, 'HP': 5000, 'Dell': 5000}.get(inputs['brand'], 0)
        
        # RAM contribution
        ram_cost = inputs['ram'] * 2000
        
        # Storage contribution
        storage_cost = inputs['storage'] * 50 if inputs['storage_type'] == 'SSD' else inputs['storage'] * 20
        
        # GPU contribution
        gpu_cost = 25000 if 'RTX 4060' in inputs['gpu'] else 15000 if 'RTX 3050' in inputs['gpu'] else 5000
        
        # Display contribution
        display_cost = (inputs['display_size'] - 13) * 2000
        
        breakdown = {
            'Base Price': base_price,
            'Brand Premium': brand_premium,
            'RAM': ram_cost,
            'Storage': storage_cost,
            'Graphics Card': gpu_cost,
            'Display': display_cost
        }
        
        return breakdown
    
    def create_comparison_chart(self, predicted_price, inputs):
        """Create comparison chart with similar laptops"""
        # Sample data for comparison (in real implementation, this would come from database)
        similar_laptops = [
            {'name': 'Similar Budget Laptop', 'price': predicted_price * 0.8, 'category': 'Budget'},
            {'name': 'Your Configuration', 'price': predicted_price, 'category': 'Selected'},
            {'name': 'Similar Premium Laptop', 'price': predicted_price * 1.2, 'category': 'Premium'},
            {'name': 'High-end Alternative', 'price': predicted_price * 1.5, 'category': 'High-end'}
        ]
        
        df_comparison = pd.DataFrame(similar_laptops)
        
        fig = px.bar(df_comparison, x='name', y='price', color='category',
                    title='Price Comparison with Similar Laptops',
                    labels={'price': 'Price (₹)', 'name': 'Laptop Configuration'})
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def create_feature_impact_chart(self, inputs):
        """Create chart showing feature impact on price"""
        features = ['RAM', 'Storage', 'Graphics Card', 'Display Size', 'Brand', 'Processor']
        
        # Simplified impact calculation
        impacts = [
            inputs['ram'] * 100,
            inputs['storage'] * 20,
            500 if 'RTX' in inputs['gpu'] else 200,
            inputs['display_size'] * 50,
            300 if inputs['brand'] in ['Apple', 'MSI'] else 100,
            400 if '13th Gen' in inputs['processor'] else 200
        ]
        
        fig = go.Figure(data=go.Bar(x=features, y=impacts, 
                                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']))
        
        fig.update_layout(
            title='Feature Impact on Price',
            xaxis_title='Features',
            yaxis_title='Price Impact (₹)',
            showlegend=False
        )
        
        return fig
    
    def create_budget_recommendations(self, budget, predicted_price):
        """Create budget-based recommendations"""
        recommendations = []
        
        if budget < predicted_price:
            difference = predicted_price - budget
            recommendations.append(f"💡 Your configuration exceeds budget by ₹{difference:,.0f}")
            recommendations.append("🔧 Consider these alternatives:")
            recommendations.append("   • Reduce RAM from current selection")
            recommendations.append("   • Choose a smaller storage capacity")
            recommendations.append("   • Select integrated graphics instead of dedicated GPU")
            recommendations.append("   • Consider a different brand")
        elif budget > predicted_price * 1.2:
            extra = budget - predicted_price
            recommendations.append(f"💰 You have ₹{extra:,.0f} extra in your budget!")
            recommendations.append("🚀 Consider these upgrades:")
            recommendations.append("   • Increase RAM for better performance")
            recommendations.append("   • Upgrade to a better graphics card")
            recommendations.append("   • Choose a larger, higher-resolution display")
            recommendations.append("   • Add more storage capacity")
        else:
            recommendations.append("✅ Your configuration fits well within your budget!")
            recommendations.append("🎯 This is a good balance of features and price.")
        
        return recommendations
    
    def main_interface(self):
        """Main Streamlit interface"""
        st.set_page_config(
            page_title="Laptop Price Predictor",
            page_icon="💻",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("💻 Laptop Price Prediction System")
        st.markdown("### Get accurate price predictions for your dream laptop configuration!")
        
        # Load model
        if not hasattr(self, 'model') or self.model is None:
            with st.spinner("Loading AI model..."):
                model_loaded = self.load_model_and_encoders()
                if not model_loaded:
                    st.error("Please ensure model files are available in the 'models' directory")
                    st.stop()
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("🎯 Configuration")
            
            # Get user inputs
            inputs = self.create_sidebar_inputs()
            
            # Budget input
            st.subheader("💰 Budget")
            budget = st.number_input("Enter your budget (₹)", 
                                   min_value=10000, max_value=500000, 
                                   value=50000, step=5000)
            
            # Predict button
            if st.button("🔮 Predict Price", type="primary"):
                with st.spinner("Analyzing configuration..."):
                    # Preprocess inputs
                    features = self.preprocess_inputs(inputs)
                    
                    # Make prediction
                    predicted_price = self.predict_price(features)
                    
                    if predicted_price:
                        st.session_state.predicted_price = predicted_price
                        st.session_state.inputs = inputs
                        st.session_state.budget = budget
        
        with col2:
            st.header("📊 Results & Analysis")
            
            if hasattr(st.session_state, 'predicted_price'):
                predicted_price = st.session_state.predicted_price
                inputs = st.session_state.inputs
                budget = st.session_state.budget
                
                # Display prediction
                st.metric("💰 Predicted Price", f"₹{predicted_price:,.0f}")
                
                # Budget comparison
                if predicted_price <= budget:
                    st.success(f"✅ Within budget! You save ₹{budget - predicted_price:,.0f}")
                else:
                    st.error(f"❌ Over budget by ₹{predicted_price - budget:,.0f}")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🔍 Comparison", "💡 Recommendations", "📋 Summary"])
                
                with tab1:
                    # Price breakdown
                    breakdown = self.create_price_breakdown(inputs, predicted_price)
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.subheader("Price Breakdown")
                        for component, cost in breakdown.items():
                            st.write(f"**{component}:** ₹{cost:,.0f}")
                    
                    with col_b:
                        # Feature impact chart
                        fig_impact = self.create_feature_impact_chart(inputs)
                        st.plotly_chart(fig_impact, use_container_width=True)
                
                with tab2:
                    # Comparison chart
                    fig_comparison = self.create_comparison_chart(predicted_price, inputs)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with tab3:
                    # Budget recommendations
                    recommendations = self.create_budget_recommendations(budget, predicted_price)
                    for rec in recommendations:
                        st.write(rec)
                
                with tab4:
                    # Configuration summary
                    st.subheader("Selected Configuration")
                    
                    config_col1, config_col2 = st.columns(2)
                    
                    with config_col1:
                        st.write(f"**Brand:** {inputs['brand']}")
                        st.write(f"**Processor:** {inputs['processor']}")
                        st.write(f"**RAM:** {inputs['ram']} GB")
                        st.write(f"**Storage:** {inputs['storage']} GB {inputs['storage_type']}")
                        st.write(f"**Graphics:** {inputs['gpu']}")
                    
                    with config_col2:
                        st.write(f"**Display:** {inputs['display_size']}\" {inputs['resolution']}")
                        st.write(f"**OS:** {inputs['os']}")
                        st.write(f"**Warranty:** {inputs['warranty']} years")
                        st.write(f"**Spec Rating:** {inputs['spec_rating']}/100")
                        st.write(f"**Predicted Price:** ₹{predicted_price:,.0f}")
            
            else:
                st.info("👈 Configure your laptop specifications and click 'Predict Price' to see results!")
                
                # Show sample predictions
                st.subheader("💡 Sample Predictions")
                sample_configs = [
                    {"name": "Budget Laptop", "specs": "Intel i3, 8GB RAM, 256GB SSD", "price": "₹35,000"},
                    {"name": "Gaming Laptop", "specs": "Intel i7, 16GB RAM, RTX 3060", "price": "₹85,000"},
                    {"name": "Premium Ultrabook", "specs": "Intel i7, 16GB RAM, 512GB SSD", "price": "₹75,000"}
                ]
                
                for config in sample_configs:
                    with st.container():
                        st.write(f"**{config['name']}**")
                        st.write(f"Specs: {config['specs']}")
                        st.write(f"Est. Price: {config['price']}")
                        st.write("---")

def run_prediction_interface():
    """Run the prediction interface"""
    interface = LaptopPricePredictionInterface()
    interface.main_interface()

if __name__ == "__main__":
    run_prediction_interface()