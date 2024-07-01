class static_class:

    columns_index_other1_variance=['sale_1', 'PFprice_SD_1', 'PFprice_CV_1', 'LSprice_average_1', 'LSprice_SD_1', 'LSprice_CV_1', 'sale_2',
     'PFprice_SD_2', 'PFprice_CV_2', 'LSprice_average_2', 'LSprice_SD_2', 'LSprice_CV_2']

    columns_index_other2_max=['total_sales', 'sale_max_t', 'sale_max', 'growth_rate_max_t',
                              'growth_rate_max_sale', 'growth_rate_max',
                            'sale_max_t_percentage', 'grow_rate_max_t_percentage',
                              'growth_rate_max_t_percentage_in_sale_max']

    columns_index_As_Asgr = ['As', 'Asgr']

    Bass_params = ['m', 'p', 'q']
    Errors_columns_name = ['MSE', 'MAE', 'RMSE', 'MAPE', 'SMAPE']
    Machine_Model_name = ['Lasso', 'ElasticNet', 'SVR', 'RT', 'KNN', 'XG']

    Machine_Model_name_used=['SVR', 'RT', 'KNN', 'XG']

    train_test_dict_name = ['train_dict', 'test_semi_dict', 'test_brand_dict']
    train_test_keys_name = ["feature", "sales", "t", "params"]
    Model_return_contend = ["sales_old_past_error", "sales_semi_past_error", "sales_predict_error_old",
                            "sales_predict_error_new", "sales_old_past_error_curve_fit",
                            "sales_semi_past_error_curve_fit", "sales_predict_error_curve_fit",
                            "sales_predict_error_new_curve_fit", "sales_predict_error_semi"]

    KNN_feature_can_predict=['PFprice_SD_1', 'PFprice_CV_1', 'LSprice_SD_1', 'LSprice_CV_1',
                             'PFprice_SD_2', 'PFprice_CV_2', 'LSprice_SD_2', 'LSprice_CV_2']

    sku_not_can_use=['泰州_雄狮(红)', "苏州_雄狮(硬)", "淮安_大红鹰(红)","泰州_大红鹰(软新品)",
                       "泰州_利群(软红长嘴)","徐州_利群(软蓝L)","南京_利群(软红长嘴)"]
    append_Macro_index = ['changes_in_cigarette_production_one_year',
                          'changes_in_cigarette_production_two_year',
                          'changes_in_Disposable_income_per_capital_one_year',
                          'changes_in_Disposable_income_per_capital_two_year',
                          'changes_in_Total_retail_sales_one_year',
                          'changes_in_Total_retail_sales_two_year']