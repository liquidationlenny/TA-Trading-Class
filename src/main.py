import Algorithms

if __name__ == '__main__':
    algos = Algorithms.Algorithms()
    algos.select_window(0, 500)
    #data = algos.bollinger_bands(plot=True)
    data = algos.grid_search_with_adx(grid_spacing=1, plot=True)
    #data = algos.keltner_channel(plot=True)
    #algos.upload_to_DB(data, 'eur_usd_keltner_channels_data', if_exists='replace')
    algos.show_DB_tables()
