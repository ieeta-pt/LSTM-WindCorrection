# LSTM-WindCorrection

**A lightweight MATLAB framework for recursive LSTM-based correction of numerical weather prediction wind speed forecasts.**


This Long Short-Term Memory (LSTM) model is designed to predict $x_t$, the forecast error, given the history of the process available up to time $t-1$, i.e, $\mathcal{F}\_{t-1} := \{ x_{t-1}, x_{t-2}, \dots, x_{t-p} \}$, where $p$ is the length of the history considered for the prediction. 

$$
 x_t = w_t - g_{t-h}^{(h)} \Leftrightarrow  w_t = g_{t-h}^{(h)} + x_t 
$$

where $w_t$ denotes the observed wind speed at time $t$, and $g_{t-h}^{(h)}$ is the GFS forecast issued at time $t-h$ for a lead time $h \in \mathbb{N}\_{0}$. The latter is fully determined at time $t$, since both $w_t$ and $g_{t-h}^{(h)}$ are available and refer to the same time instant. Thus, the corrected GFS forecast is defined as

$$
\begin{equation}
    \hat{w}_t^{(h)} = g_t^{(h)} + \tilde{x}_t^{(h)}
\end{equation}
$$

where $\tilde{x}_t^{(h)}$ denotes an estimate of the unknown future forecast error. In this work a correction composed of two components is considered

$$
\begin{equation}
    \tilde{x}_t^{(h)} = \bar{m}_{t} + \hat{x}_{t}^{(h)}
\end{equation}
$$

where $\bar{m}\_{t}$ is a mean bias correction term and $\hat{x}_{t}^{(h)}$ is the forecast error predicted by the LSTM model. The bias correction term is defined as

$$
\begin{equation}
    \bar{m}_t=\frac{1}{t}\sum_{i=1}^{t}x_i,
\end{equation}
$$

corresponding to the average of past forecast errors available up to time $t$.


Therefore, the LSTM model predicts $x_t$ given the information in $\mathcal{F}_{t-1}$, or mathematically,

$$
\hat{x}_t = f(\mathcal{F}_{t-1})
$$

where $f(.)$ is a general function that expresses the operations of the LSTM model. 

To predict for $h = 1, 2, \dots$ steps ahead, the model is adapted to predict $x_{t+h}$ by recursively feeding its own predictions back into the input for each subsequent time step, i.e.,

$$
\hat{x}_{t}^{(h)} = f(\mathcal{F}_{t-1+h}),
$$

where $\mathcal{F}\_{t-1+h} := \{\hat{x}\_{t-1+h}, \dots, \hat{x}\_t, x_{t-1}, x_{t-2}, \dots, x_{t-p-h}\}$ is the input history for future steps, including previous predictions $\hat{x}$ and original observed data $x$. Since the model is sequence-to-one, it receives as input the vector $\mathcal{F}_{t-1}$ of length $p$ and predicts a single continuous value, denoted as $x_t$.  The fundamental component in this architecture is the LSTM cell, which consists of three gates, namely forget $f$, input $i$ and output $o$, summarized as follows

$$
\begin{align}
    f_t &= \sigma \big(W_f [h_{t-1}, x_t] + b_f\big), \nonumber \\
    i_t &= \sigma \big(W_i [h_{t-1}, x_t] + b_i\big), \nonumber \\
    \tilde{C}_t &= \tanh \big(W_C [h_{t-1}, x_t] + b_C\big), \nonumber \\
    C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, \nonumber \\
    o_t &= \sigma \big(W_o [h_{t-1}, x_t] + b_o\big), \nonumber \\
    h_t &= o_t \odot \tanh(C_t), \nonumber
\end{align}
$$

where $\sigma$ denotes the sigmoid function, $W_{\cdot}$ and $b_{\cdot}$ are weight matrices and bias vectors, respectively, and $\odot$ denotes element-wise multiplication.
The model outputs the predicted value for $x_t$ and is computed from the final hidden state $h_t$ as

$$
\hat{x}_t = W_y h_t + b_y,
$$

where $W_y$ and $b_y$ are weights and biases of the fully connected layer. The referred $W_{\cdot}$ matrices and $b_{\cdot}$ vectors are trainable parameters of the LSTM model.


This framework has been implemented in MATLAB, and synthetic data is provided along to illustrate how the model is employed.


**To properly cite this work use:** Gomes V, Martins A, Carvalho D, Gouveia S. (2026) Recursive Multi-step-ahead LSTM Correction of GFS Wind Speed Forecasts in Portugal. Pattern Analysis and Applications.
