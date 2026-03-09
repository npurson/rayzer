rayzer/ 是当前基于 rayzer codebase 的训练框架，rayzer/submodules/vggt 有 vggt 的网络代码，xfactor/ 有 xfactor 的部分代码

1. xfactor 设计了这样一件事：
对于两帧的 context view 和 target view 输入，采用两个不重叠的 mask 分别将他们划分为 aug a 和 aug b。pose encoder 看到 context aug a 和 target aug a，随后通过 render 输入 context aug b 和 pose，来预测 target aug b。

2. 我的想法是这样的：context 划分 aug a 和 aug b 这里是没有必要的，而且他这种把 pose encoder 独立出来的做法不符合现在的通用模型范式。所以我想做：
1. 基于通用的 VGGT 架构，但是在它的 global attention 层加入多级掩码来实现类似的事
2. 仿照 xfactor 读取两帧图像作为输入，只对 target view 划分 aug a 和 aug b
3. 在 vggt 中，context view 只能看到自身，target view aug a + pose token 1 能看到它们自身以及 context view，target view aug b + pose token 2 能看到他们自身以及 target view aug a 和 context view。
4. pose token 不采用 xfactor 的完全隐式表示，而是还是采用 rayzer 这样解码成 se3 的方式。
5. 另有一个 decoder，按照 rayzer 的方式将 se3 转成 plucker，并用上 context view 一起预测 target aug b 前面**经过 vggt 后的 detached 的特征**。
6. pose token2 的训练目标是对齐 detached 的pose token 1 的特征。
7. pose token1、2 来源于同一个 vggt 里本身的 pose embedding，但是在训练的时候复制两份，推理的时候就不用复制也不需要mask，直接采用
8. 以上所说的请在 rayzer/ 里实现，命名为 spa3r（Spa3R），也要能使用原有的脚本评测 psnr 和 pose。

另外，请不要试图运行 python 测试等，本设备上没有 python 运行环境，你写好后我会传到其他设备运行。

现在，请你看下你对我的设计或者背后原因理解有没有不清楚的地方，在沟通完毕后请开始实现
