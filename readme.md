# Neuromatch_Academy_Afrovenator ~ By the nerdy vampires

- Analysis of **Joystick Track data** from Miller 2008. In this 3 week project, we intended to determine how far into the future cursor position can be predicted with a reasonable accuracy from ECoG signals. This could be used to compensate the latency in BCI applications.

- This project was conducted as part of the <a href="https://academy.neuromatch.io/" target="_blank">Neuromatch Academy program</a> 2023

# Installation
You can create an conda environment
```
conda env create -f environment.yml -n neuromatch
conda activate neuromatch
```

# Run experiments
Every experiment in this project is separated in modules:
* data_preprocessing.py → Preprocessing
* channel_selection.py → ChannelSelection
* regression_models.py → RegressionModels


The main function to run analysis is given in `run_full_analysis.py` and is decomposed in loading, preprocessing, channel selection per brodmann areas and linear regression.
You can run a full analysis with specific parameters using the following command: 

```
python run_full_analysis.py --feature [FEATURE] --window_size [WINDOW_SIZE] --num_points [NUM_POINTS] --brodmann_area_number [BRODMANN_NUMBER] --frequency_band [MIN_FREQ] [MAX_FREQ] --participant_ind [PARTICIPANT_IND] --forc_dist [FORECASTING_DISTANCE]
```

```
def run_analysis(args):
    '''Run full analysis using a set of parameters'''

    # Load data
    alldat = load_data()

    feature = args.feature
    window_size = args.window_size
    num_points = args.num_points
    brodmann_area_number = args.brodmann_area_number
    frequency_band = args.frequency_band
    participant_ind = args.participant_ind
    forc_dist = args.forc_dist
    random_seed = args.random_seed

    assert forc_dist < MAX_OFFSET
    
    #Select one participant
    dat = alldat[0,participant_ind]
    label = get_label(feature, dat, MAX_OFFSET)

    # Preprocess the data

    preprocessed_data = preproc_fun.car(dat["V"])
    preprocessed_data = preproc_fun.band_pass(preprocessed_data, frequency_band[1],frequency_band[0], order = 5)


    # Select channels

    channel_sel_data = channel_sel.select_by_Brodmann(preprocessed_data, brodmann_area_number, dat["Brodmann_Area"])
    if channel_sel_data.shape[1] > 3:
        channel_sel_data, ev = channel_sel.channel_pca(channel_sel_data, num_components = 3, get_explained_variance = True)

    # Run regression models

    X = regression_models.create_design_matrix(channel_sel_data, window_size = window_size, num_points = num_points, forc_dist = forc_dist, max_offset = MAX_OFFSET)
    print(X.shape, label.shape)
    X_train ,X_test, y_train, y_test = regression_models.temporal_split(X, label, test_size=0.2, seed = random_seed)
    y_pred, train_score, test_score, coeffs, intercept = regression_models.linear_regression(X_train, X_test, y_train, y_test, return_all=True)

    # Save score and save parameters
    save_scores(train_score, test_score, args, datetime.now().strftime("%m/%d/%Y,%H:%M:%S"), filename = "search_results.csv")

```
## Data Analysis
The main module for data analysis is located in `analysis.py`. It consists in 2 steps 
1) Plot the $R^2$ score against the window size with a forecasting distance of 0
2) Plot the $R^2$ score against the forecasting distance for a fixed window size

All functions are stored in
* analysis.py → DataAnalysis


## Collaborators
Aditya Narendra, Andy Bonnetto, Chayanon Kitkana, Paola Juárez, Ruman Ahmed Shaikh & Taima Crean


