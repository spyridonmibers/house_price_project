import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка страницы
st.set_page_config(
    page_title="Предсказание цены дома",
    page_icon="🏠",
    layout="wide"
)

# Заголовок
st.title("🏠 Предсказание цены дома")
st.markdown("Введите данные о доме или загрузите CSV-файл для предсказания")

# ============================================
# Загрузка модели
# ============================================
@st.cache_resource
def load_model():
    """Загрузка обученной модели CatBoost"""
    model_path = "house_price_predictor/catboost_model.cbm"
    
    if os.path.exists(model_path):
        try:
            model = CatBoostRegressor()
            model.load_model(model_path)
            st.success("✅ Модель успешно загружена!")
            return model
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            return None
    else:
        st.error("❌ Файл модели catboost_model.cbm не найден!")
        st.info("💡 Убедитесь, что модель сохранена в той же директории, что и app.py")
        return None

# ============================================
# Функции для предобработки
# ============================================
def preprocess_data(df):
    """Предобработка данных (копия вашей логики)"""
    df = df.copy()
    
    # Числовые колонки для заполнения нулями
    num_cols = ['LotFrontage', 'OverallQual', 'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2',
                'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'PoolArea', 'MiscVal']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Категориальные колонки
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in ['Id', 'SalePrice']]
    
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # Обработка GarageYrBlt
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: 1 if x != 0 else 0)
    
    # Создаём новые признаки
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['Fresh_remod'] = df['YrSold'] - df['YearRemodAdd']
    else:
        df['Fresh_remod'] = 0
    
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['age_Bild'] = df['YrSold'] - df['YearBuilt']
    else:
        df['age_Bild'] = 0
    
    # Удаляем ненужные колонки
    cols_to_drop = ['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold', 'Id']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    return df

# ============================================
# Ручной ввод данных (сокращённая версия)
# ============================================
def manual_input_form():
    """Форма для ручного ввода данных"""
    st.subheader("📝 Ручной ввод данных")
    
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📍 Основные характеристики**")
            lot_area = st.number_input("LotArea (площадь участка, кв.футы)", min_value=0, value=10000)
            overall_qual = st.slider("OverallQual (общее качество)", 1, 10, 6)
            overall_cond = st.slider("OverallCond (общее состояние)", 1, 10, 5)
            year_built = st.number_input("YearBuilt (год постройки)", min_value=1872, max_value=2010, value=2000)
            gr_liv_area = st.number_input("GrLivArea (жилая площадь, кв.футы)", min_value=0, value=1500)
            total_bsmt_sf = st.number_input("TotalBsmtSF (площадь подвала)", min_value=0, value=800)
            full_bath = st.number_input("FullBath (полные ванные)", min_value=0, max_value=4, value=2)
            bedrooms = st.number_input("BedroomAbvGr (спальни)", min_value=0, max_value=5, value=3)
            
        with col2:
            st.markdown("**🏠 Дом и гараж**")
            garage_cars = st.number_input("GarageCars (машиномест)", min_value=0, max_value=4, value=2)
            garage_area = st.number_input("GarageArea (площадь гаража)", min_value=0, value=400)
            fireplaces = st.number_input("Fireplaces (камины)", min_value=0, max_value=3, value=0)
            central_air = st.selectbox("CentralAir", ['Y', 'N'])
            electrical = st.selectbox("Electrical", ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'])
            neighborhood = st.selectbox("Neighborhood", 
                                       ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 
                                        'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer'])
        
        submitted = st.form_submit_button("🔮 Предсказать цену", use_container_width=True)
        
        if submitted:
            data = {
                'MSSubClass': [60],
                'MSZoning': ['RL'],
                'LotFrontage': [70],
                'LotArea': [lot_area],
                'Street': ['Pave'],
                'Alley': ['None'],
                'LotShape': ['Reg'],
                'LandContour': ['Lvl'],
                'Utilities': ['AllPub'],
                'LotConfig': ['Inside'],
                'LandSlope': ['Gtl'],
                'Neighborhood': [neighborhood],
                'Condition1': ['Norm'],
                'Condition2': ['Norm'],
                'BldgType': ['1Fam'],
                'HouseStyle': ['2Story'],
                'OverallQual': [overall_qual],
                'OverallCond': [overall_cond],
                'YearBuilt': [year_built],
                'YearRemodAdd': [year_built],
                'RoofStyle': ['Gable'],
                'RoofMatl': ['CompShg'],
                'Exterior1st': ['VinylSd'],
                'Exterior2nd': ['VinylSd'],
                'MasVnrType': ['None'],
                'MasVnrArea': [0],
                'ExterQual': ['TA'],
                'ExterCond': ['TA'],
                'Foundation': ['PConc'],
                'BsmtQual': ['TA'],
                'BsmtCond': ['TA'],
                'BsmtExposure': ['No'],
                'BsmtFinType1': ['Unf'],
                'BsmtFinSF1': [0],
                'BsmtFinType2': ['Unf'],
                'BsmtFinSF2': [0],
                'BsmtUnfSF': [0],
                'TotalBsmtSF': [total_bsmt_sf],
                'Heating': ['GasA'],
                'HeatingQC': ['Ex'],
                'CentralAir': [central_air],
                'Electrical': [electrical],
                '1stFlrSF': [gr_liv_area],
                '2ndFlrSF': [0],
                'LowQualFinSF': [0],
                'GrLivArea': [gr_liv_area],
                'BsmtFullBath': [0],
                'BsmtHalfBath': [0],
                'FullBath': [full_bath],
                'HalfBath': [1],
                'BedroomAbvGr': [bedrooms],
                'KitchenAbvGr': [1],
                'KitchenQual': ['TA'],
                'TotRmsAbvGrd': [7],
                'Functional': ['Typ'],
                'Fireplaces': [fireplaces],
                'FireplaceQu': ['None'] if fireplaces == 0 else ['TA'],
                'GarageType': ['Attchd'],
                'GarageYrBlt': [2000],
                'GarageFinish': ['Unf'],
                'GarageCars': [garage_cars],
                'GarageArea': [garage_area],
                'GarageQual': ['TA'],
                'GarageCond': ['TA'],
                'PavedDrive': ['Y'],
                'WoodDeckSF': [0],
                'OpenPorchSF': [0],
                'EnclosedPorch': [0],
                '3SsnPorch': [0],
                'ScreenPorch': [0],
                'PoolArea': [0],
                'PoolQC': ['None'],
                'Fence': ['None'],
                'MiscFeature': ['None'],
                'MiscVal': [0],
                'MoSold': [6],
                'YrSold': [2008],
                'SaleType': ['WD'],
                'SaleCondition': ['Normal']
            }
            
            df_input = pd.DataFrame(data)
            return df_input
    
    return None

# ============================================
# Основное приложение
# ============================================
def main():
    # Выбор режима
    mode = st.radio(
        "Выберите способ ввода данных:",
        ["📂 Загрузить CSV файл", "✍️ Ввести данные вручную"],
        horizontal=True
    )
    
    df_input = None
    
    if mode == "📂 Загрузить CSV файл":
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл загружен! {df_input.shape[0]} строк, {df_input.shape[1]} колонок")
            st.dataframe(df_input.head())
    
    else:
        df_input = manual_input_form()
    
    # Предсказание
    if df_input is not None:
        st.divider()
        st.subheader("🔮 Результат предсказания")
        
        # Предобработка
        with st.spinner("🔄 Обработка данных..."):
            df_processed = preprocess_data(df_input)
        
        # Загрузка модели
        model = load_model()
        
        if model is not None:
            with st.spinner("🤖 Выполняется предсказание..."):
                predictions = model.predict(df_processed)
                predictions = np.exp(predictions)
                
                # Отображение результатов
                for i, pred in enumerate(predictions):
                    st.metric(
                        label=f"🏠 Предсказанная цена дома {i+1}",
                        value=f"${pred:,.2f}"
                    )
                
                # Скачивание результатов
                if mode == "📂 Загрузить CSV файл" and len(predictions) > 0:
                    result_df = pd.DataFrame({
                        'Id': df_input.get('Id', range(1, len(predictions) + 1)),
                        'SalePrice': predictions
                    })
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Скачать предсказания в CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        else:
            st.error("❌ Модель не загружена!")

if __name__ == "__main__":
    main()
