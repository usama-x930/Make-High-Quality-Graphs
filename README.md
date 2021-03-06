# Make High Quality Graphs
 Make beautiful, high quality graphs uing matplotlib and seaborn

## Requirements
* Python >= 3.4
* Matplotlib
* Seaborn
* Numpy
* Pandas

## Generating HD quality graphs

Include this code snippet in you plotting code and generate HD quality graphs.

```
%config InlineBackend.figure_format = 'retina'
plt.style.use('fivethirtyeight')
```

I am using "fivethirtyeight" stylesheet. Have a look at this.
```
https://matplotlib.org/3.1.1/gallery/style_sheets/fivethirtyeight.html
```


## Activation Function Plots

<p align="center">
  <img src="Figs/relu.png" width="300" title="hover text">
  <img src="Figs/sigmoid.png" width="310" title="hover text">
 <img src="Figs/leaky_relu_plot.jpg" width="300" title="hover text">
</p>

## Bar Charts
<p align="center">
  <img src="Figs/ch12.png"  width="500" title="hover text">
  <img src="Figs/hist3.png"  width="500" title="hover text">
 <img src="Figs/grp.png"  width="500" title="hover text">
 <img src="Figs/AP2.png" width="320" title="hover text">
 <img src="Figs/Picture1.png" width="320" title="hover text">
</p>

## Precision, Recall, Accuracy, Loss, F1-Score

Run
``` 
python plots.py
```

<p align="center">
  <img src="images/accuracy.png"  width="300" title="hover text">
  <img src="images/loss.png"  width="300" title="hover text">
 <img src="images/precision.png"  width="300" title="hover text">
 <img src="images/recall.png" width="300" title="hover text">
 
</p>
