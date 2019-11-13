tickers = c("AABA","ADBE","ADSK","AKAM","ALXN","AMD","AMZN","AN","AZO","BIIB","BRK.B","BSX","CBG","CELG","CERN","CHK","CHTR","CMG","CNC","CRM","CTXS","CXO","DISCA","DISCK","DISH","DLTR","DVA","EA","EBAY","ESRX","ETFC","EVHC","EW","FB","FCX","FFIV","FISV","FTI","GOOG","GOOGL","HCA","HOLX","HSIC","IDXX","ILMN","INCY","ISRG","IT","KMX","KORS","LH","LKQ","LVLT","MHK","MNK","MNST","MTD","MU","MYL","NFLX","NFX","ORLY","PCLN","PWR","PYPL","QRVO","REGN","RHT","RIG","SNPS","SRCL","TDC","TDG","TRIP","UA","UAA","UAL","ULTA","URI","VAR","VRSK","VRSN","VRTX","WAT")
historical = BatchGetSymbols::BatchGetSymbols(tickers=tickers, first.date="2000-01-1", last.date="2019-01-01", freq.data='daily', thresh.bad.data=0.95)

for (ticker in historical$df.control$ticker){
  action = historical$df.tickers[historical$df.tickers$ticker == ticker, ]
  for (year in 2019:2000){
    returns = action[action$ref.date >= sprintf("%s-01-01", year) & action$ref.date <= sprintf("%s-12-31", year),]$ret.adjusted.prices
    if (length(returns) == 0) next
    if (nortest::lillie.test(returns)$p.value >= 0.05 && length(returns) >= 250){
      hist(returns, main=sprintf("%s %s", ticker, year))
      jarque = tseries::jarque.bera.test(returns)$p.value
      shapiro = shapiro.test(returns_AABA_2019)$p.value
      print("jarque=%f shapiro=%f", jarque, shapiro)
      line = readline(prompt="$ ")
      if (line=="exit") break
    }
  }
  if (line=="exit") break
}

AABA_2019 = BatchGetSymbols::BatchGetSymbols(tickers="AABA", first.date="2018-01-01", last.date="2018-12-31", freq.data='daily', thresh.bad.data=0.95)$df.tickers
returns_AABA_2019 = AABA_2019[AABA_2019$ref.date >= "2018-01-03",]$ret.adjusted.prices

tseries::jarque.bera.test(returns_AABA_2019)$p.value
shapiro.test(returns_AABA_2019)$p.value

length(returns_AABA_2019)
ggpubr::ggdensity(returns_AABA_2019)
ggpubr::ggqqplot(returns_AABA_2019)

write.csv(AABA_2019, "~/AABA_2019.csv")