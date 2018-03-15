import funcs.iofunctions as file_funs

file_trip = "trip.csv"  #"c:/_reasearch/trip.csv"
file_station = "station.csv"
file_weather = "weather.csv"
file_list = [file_trip, file_station, file_weather]
Trip, Station,Weather = file_funs.get_df_from_files(file_list)
