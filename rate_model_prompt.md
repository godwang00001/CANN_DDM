# 离散 Edge（Sigmoid）稳态的 Mexican-hat Connectivity 构造

## —— 问题背景、结构约束与自洽实现说明

---

## 0. 背景与目标

我们考虑一类 firing-rate 网络（连续/离散皆可）：

$$

\dot{\mathbf r} = -\mathbf r + \phi(\mathbf W \mathbf r + \mathbf I_{\text{clamp}})

$$

其中：

- $\mathbf r \in [0,1]^N$ 为神经元放电率；
- $\mathbf W \in \mathbb R^{N\times N}$ 为需要构造的连接矩阵；
- $\phi(\cdot)$ 选为带仿射的 sigmoid：
$$
\phi(u)=\sigma(\alpha u+\beta),\qquad 
\sigma(z)=\frac{1}{1+e^{-z}}
$$
- $\mathbf I_{\text{clamp}}$ 是仅对边界区域施加的外加输入，用于把边界神经元强制固定到 $0/1$（吸收边界）。

假设

**目标：**  
在有限神经空间 $\theta\in[-\pi/2,\pi/2]$ 上，构造一个离散 connectivity $\mathbf W$，使得如下 **edge-shaped profile**

$$
r_{\text{target}}(\theta;\theta')=\sigma\big(\gamma(\theta-\theta')\big)
$$

是系统的稳定稳态，并且：

- 在 interior 区域严格满足 $r_{\text{pred}}\approx r_{\text{target}}$；
- 误差与边界伪影被系统性地推到远离 interior 的区域；
- edge 的“有效存在区间”是 **可计算、可验证、可回退的**，而非经验设定。

---

## 1. 核心思想回顾

我们采用如下结构性分区：

1. **interior（严格内区）**
  - 不允许饱和；
  - sigmoid 可近似线性；
  - 用于解析匹配 $\mathbf W\mathbf r \sim \theta$。
2. **buffer / transition 区**
  - 允许一定误差；
  - 用于吸收有限尺寸与边界效应。
3. **clamp 区（吸收边界）**
  - 通过 $\mathbf I_{\text{clamp}}$ 强制推入 $0/1$ 饱和；
  - 不参与 $\mathbf W$ 的解析匹配。

关键升级在于：  
👉 **interior 与 clamp 的边界不再“人为给定”，而是由目标解与误差容限共同自洽确定。**

---

## 2. 自动确定 interior 阈值 $\theta_1$

### 2.1 设计原则

interior 的定义不应依赖人为输入，而应由目标解本身决定。

我们要求 interior 中的神经元活动严格落在 sigmoid 的“可工作区间”：

$$
r_{\text{target}}(\theta)\in[\varepsilon,1-\varepsilon]
$$

### 2.2 自洽定义（不作为输入参数）

因此，$\theta_1$ 定义为 **目标解第一次触及 boundary 的位置**：

 $$
\boxed{
\theta_1

\min\Big
\theta>0:
r_{\text{target}}(\theta)=1-\varepsilon
\Big
}
$$

对 sigmoid edge，

$$

r_{\text{target}}(\theta)=\sigma(\gamma\theta)
\quad\Rightarrow\quad
\theta_1

\frac{1}{\gamma}
\log\frac{1-\varepsilon}{\varepsilon}.
$$

因此：

- $\theta_1$ **完全由 $(\gamma,\varepsilon)$ 决定**；
- 不再作为函数或算法的自由输入；
- interior 的定义从“人为设定”升级为“结构推导”。

---

## 3. interior 拟合与 connectivity 构造（不变）

在

$$
\theta\in[-\theta_1,\theta_1]
$$

内，目标解远离饱和，sigmoid 可近似线性：

$$
r \approx \alpha(\mathbf W r) + \beta
$$

从而可以通过线性回归 / 匹配的方法，构造 Mexican-hat（DoG）型 Toeplitz kernel：

- 兴奋短程、抑制长程；
- 行和近零（避免全局漂移）；
- 在 interior 内满足
$$
\alpha(\mathbf W r_{\text{target}})(\theta)+\beta
\approx
\gamma\theta.
$$

---

## 4. clamp 阈值 $\theta_2$ 的**可行性验证与自适应回退**

### 4.1 初始设定

$\theta_2$ 仍可作为一个 **候选 clamp 起始位置** 给定，例如接近边界：

$$
\theta_2^{(0)} \lesssim \frac{\pi}{2}.
$$

但与旧方案不同的是：

> **$\theta_2$ 不再被假设为可行，而必须通过误差验证。**

---

### 4.2 误差一致性检验

构造好 $\mathbf W$ 与 $\mathbf I_{\text{clamp}}$ 后，计算稳态解 $r_{\text{pred}}(\theta)$，并检查：

## $$

\boxed{
\left|
r_{\text{pred}}(\theta_2)

r_{\text{target}}(\theta_2)
\right|
\le
\varepsilon
}
$$

含义是：

- 在 clamp 起始点，edge 尚未被边界强制“拉断”；
- interior 与 buffer 的误差没有侵蚀到该位置。

---

### 4.3 自适应回退机制（关键升级）

若上述条件 **不满足**，则说明：

> 当前选择的 $\theta_2$ 过大，edge 在到达该位置前已经被边界效应破坏。

此时应：

1. 沿着神经索引向 interior 方向搜索；
2. 定义

# $$

\boxed{
\theta_2^

\max\Big
\theta>\theta_1:
\big|
r_{\text{pred}}(\theta)-r_{\text{target}}(\theta)
\big|\le\varepsilon
\Big
}
$$

1. **自动更新**

$$
\theta_2 \leftarrow \theta_2^
$$

并在实现中：

- 显式打印 /记录这一回退：
  > `theta_2 reduced from θ2_init to θ2* due to boundary-induced error`
- 将其作为该参数组合下 **edge 可维持的最大 clamp 阈值**。

---

## 5. edge 有效长度的自洽定义

在新的自洽框架下，edge 稳态的有效存在长度定义为：

# $$

\boxed{
L_{\text{edge}}

2(\theta_2^-\theta_1)
}
$$

其物理含义为：

- interior 由目标解与 $\varepsilon$ 自动确定；
- clamp 起点由动力学误差反向约束；
- $L_{\text{edge}}$ 成为一个 **模型输出量**，而非输入假设。

---

## 6. 方法论总结（升级要点）

相较于旧版本，本更新实现了三点关键提升：

1. **$\theta_1$ 不再是调参量，而是由目标解解析确定**；
2. **$\theta_2$ 不再被假设可行，而是必须通过误差验证**；
3. edge 的“可维持范围”成为
  > **由目标解 + connectivity + 动力学共同决定的可计算对象**。

这使得“edge effect”从数值经验现象，升级为：

> **可推导、可报错、可回退、可比较的结构性质。**

---

## 7. 与实现接口的对应关系（供代码参考）

在实际函数实现中：

- 函数输入：
  - $N,\gamma,\alpha,\beta,\theta_2^{(0)},\varepsilon$
- 内部自动计算：
  - $\theta_1$（由 crossing 决定）
  - 若需要，$\theta_2^$（由误差回退）
- 函数输出：
  - $\mathbf W$
  - $r_{\text{pred}}, r_{\text{target}}$
  - $(\theta_1,\theta_2^,L_{\text{edge}})$ 作为诊断量

---

