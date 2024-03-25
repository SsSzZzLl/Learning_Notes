# Decision Tree Model: Supervised Learning Classification Algorithm

$\textcolor{orange} {Entropy}$ —— represents the degree of confusion and uncertainty of an object

Entropy function expression :

$$
Entorpy = -\sum_{i = 1}^{n} p(x_i)\log_2p(x_i)
$$

When type A and type B each account for 50% probability : 

$$
Entorpy = -[\frac{1}{2}\log_2(\frac{1}{2}) + \frac{1}{2}\log_2(\frac{1}{2})]= -(-\frac{1}{2} - \frac{1}{2}) = 1
$$

When there is only type A or type B :

$$
Entorpy = -[1 * \log_21 + 0] = -(0 + 0) = 0
$$

When the entropy is maximum 1, it is the state with the worst classification effect. On the contrary, when it is minimum 0, it is the state of complete classification.

$\textcolor{orange}{Information\ Gain}$ —— The measurement method of describing the division effect of test conditions is the difference between the information entropy of the parent node and the weighted average of the information entropy of each leaf node.

$$
Gain(D, a) = Ent(D) - \sum_{i = 1}^{v}\frac{|D_v|}{|D|}Ent(D_v)
$$

$Ent(D)$is the impurity of the parent node, $\sum_{i = 1}^{v}\frac{|D_v|}{|D|}Ent(D_v)$is the weighted average of the impurity of each branch node under the support of each weight

$\textcolor{orange}{Information\ Gain\ Rate}$ :

$$
Gain_{ratio}(D, A) = \frac{Gain(D, A)}{IV(A)}
$$

$$
IV(A) = - \sum_{i = 1}^{v}\frac{|D_v|}{|D|}\log_2\frac{|D_v|}{|D|}
$$

IV(A) is called intrinsic value, actually is the purity of attribute A