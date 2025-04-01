from summary import summary

#usage
if __name__=="main":
  print("Summary: ",summary(
    file_name = "bowlero",
    percentile=90,
    dynamic_threshold=50,
    dynamic_adjustment=True
    ))
