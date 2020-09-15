from datetime import datetime
timestamp = 1508158113
dt_object = datetime.fromtimestamp(timestamp)
date = str(dt_object).split(' ')[0]
print("date =", date)
print("type(dt_object) =", type(dt_object))