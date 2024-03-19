import ruptures as rpt
import statsmodels.api as sm
import numpy as np
import plotly.express as px


def get_segm(i):
    df = frames[i]
    subset = df[df[f'{i}'] > .12]
    temp = subset.tweetcreatedts.diff()
    intervals = temp.values.astype('float')
    model = 'rbf'
    algo = rpt.Pelt(model=model)
    result = algo.fit_predict(intervals[1:], pen=4)
    subsections = []
    prev = 0
    for k in result:
        subsections.append(range(prev,k))
        prev = k
    # subsections
    splitter = np.argmin(np.abs([np.mean(s) for s in np.split(intervals[1:], result[:-1])]))#np.argmin([np.mean(s) for s in np.split(intervals[1:], result[:-1])])
    splitter = subsections[splitter][0]
    subset['ones'] = 1
    # subset['splitter'] = (subset['tweetcreatedts'] >= pd.to_datetime(subset.tweetcreatedts.values[splitter])).astype(int)
    T = subset.groupby(pd.Grouper(freq = "H", key = "tweetcreatedts")).ones.sum()
    fig = px.bar(T.reset_index(), x = 'tweetcreatedts', y= 'ones')
    T = T.to_frame()
    T['split'] = T.index >= (pd.to_datetime(subset.tweetcreatedts.values[splitter])) - pd.Timedelta(hours = 1)
    T = T.reset_index()
    # px.line(T, x = 'tweetcreatedts', y= 'ones', color = 'split' )
    T['time'] = T.split.cumsum() - 1
    T['const'] = 1
    to_fit = T[T.split]
    model = sm.Poisson(to_fit.ones, sm.add_constant(to_fit.time)).fit()
    if sum(~T.split) <= 1:
        raise Exception("Intervals wrong")
    norm_rate = T[~T.split].ones
    base_model = sm.Poisson(norm_rate, np.ones_like(norm_rate)).fit()
    b = model.params[0]
    a = model.params[1]
    B = base_model.params[0]
    Bmm = np.mean(norm_rate)
    if B > b or a > 0:
        raise Exception("Model did not fit")
    n = int((B - b) / (a)) + 1
    preds = model.predict(sm.add_constant(np.arange(n)))
    relinds = subset[subset.tweetcreatedts > (pd.to_datetime(subset.tweetcreatedts.values[splitter]))]
    return [i, b, a, B, sum(preds), model.prsquared, True], model, relinds.index.to_list()