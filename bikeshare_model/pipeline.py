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

    ('Weekday_Imputation', WeekdayImputer(config.training_config.weekdayimputer_var)),
    ('Weathersit_Imputation', WeathersitImputer(config.training_config.weathersitimputer_var)),
    ##==========Mapper======##
      ('map_yr',Mapper(config.training_config.yr_var, config.training_config.yr_mappings)),
        ('map_mnth',Mapper(config.training_config.mnth_var,config.training_config.mnth_mappings)),
          ('map_season',Mapper(config.training_config.season_var, config.training_config.season_mappings)),
            ('map_weathersit',Mapper(config.training_config.weathersit_var, config.training_config.weathersit_mappings)),
               ('map_holiday',Mapper(config.training_config.holiday_var, config.training_config.holiday_mappings)),
                ('map_workingday',Mapper(config.training_config.workingday_var, config.training_config.workingday_mappings)),
                  ('map_hr',Mapper(config.training_config.hr_var, config.training_config.hr_mappings)),
     # Handling Outlier of temp,atemp,hum,windspeed
    ('handle_outliers_temp',OutlierHandler(config.training_config.handle_outliers_temp_var)),
    ('handle_outliers_atemp',OutlierHandler(config.training_config.handle_outliers_atemp_var)),
    ('handle_outliers_hum',OutlierHandler(config.training_config.handle_outliers_hum_var)),
    ('handle_outliers_windspeed',OutlierHandler(config.training_config.handle_outliers_windspeed_var)),

    #OneHot Encode 'weekday' column
    ('OneHot_Encode_weekday',WeekdayOneHotEncoder(config.training_config.weekdayimputer_var)), 
       
    # scale
    ('scaler', StandardScaler()),
    ('model_rf', RandomForestRegressor(n_estimators=150, max_depth=5,random_state=42))

])