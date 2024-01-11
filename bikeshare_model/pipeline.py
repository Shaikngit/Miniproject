import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder 

bikeshare_pipeline=Pipeline([

    ('Weekday_Imputation', WeekdayImputer(config.model_config_data.weekdayimputer_var)),
    ('Weathersit_Imputation', WeathersitImputer(config.model_config_data.weathersitimputer_var)),
    ##==========Mapper======##
      #  ('map_yr',Mapper(config.model_config_data.yr_var, config.model_config_data.yr_mappings)),
        ('map_yr',Mapper('yr',{2011: 0, 2012: 1})),
        ('map_mnth',Mapper(config.model_config_data.mnth_var,config.model_config_data.mnth_mappings)),
          ('map_season',Mapper(config.model_config_data.season_var, config.model_config_data.season_mappings)),
            ('map_weathersit',Mapper(config.model_config_data.weathersit_var, config.model_config_data.weathersit_mappings)),
              ('map_holiday',Mapper(config.model_config_data.holiday_var, config.model_config_data.holiday_mappings)),
                ('map_workingday',Mapper(config.model_config_data.workingday_var, config.model_config_data.workingday_mappings)),
                  ('map_hr',Mapper(config.model_config_data.hr_var, config.model_config_data.hr_mappings)),
     # Handling Outlier of temp,atemp,hum,windspeed
    ('handle_outliers_temp',OutlierHandler(config.model_config_data.handle_outliers_temp_var)),
    ('handle_outliers_atemp',OutlierHandler(config.model_config_data.handle_outliers_atemp_var)),
    ('handle_outliers_hum',OutlierHandler(config.model_config_data.handle_outliers_hum_var)),
    ('handle_outliers_windspeed',OutlierHandler(config.model_config_data.handle_outliers_windspeed_var)),

    #OneHot Encode 'weekday' column
    ('OneHot_Encode_weekday',WeekdayOneHotEncoder(config.model_config_data.weekdayimputer_var)), 
       
    # scale
    ('scaler', StandardScaler()),
    ('model_rf', RandomForestRegressor(n_estimators=150, max_depth=5,random_state=42))

])