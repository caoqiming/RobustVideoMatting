# speech script

## 2

So, what exactly is Video Matting？
In simple terms, it's the process of precisely extracting a foreground object from a video sequence. We do this by calculating what we call an Alpha Matte, which is essentially a transparency map.
Technically, we look at every single pixel as a mix of the foreground and the background. You can see the math here:

Now, you might wonder: how is this different from regular segmentation?
Well, while segmentation usually creates a rough, binary 'cut-out,' matting is much more detailed. It handles those tricky 'half-transparent' areas—think of things like fine strands of hair, smoke, or motion blur.

Finally, because we’re working with video, we have to focus on Temporal Consistency. This just means we make sure the foreground stays smooth and flicker-free as the video plays from one frame to the next."

## 3

For our core technical solution, I have selected the paper titled 'Robust High-Resolution Video Matting with Temporal Guidance,' published by Lin et al. at WACV 2022.

You might ask, why did I choose this specific scheme over other existing methods?

1. Superior Handling of High-Resolution Details Many traditional segmentation methods treat the image as a binary cut-out, which often results in jagged edges. As seen in our examples here, this method excels at Robust High-Resolution processing. It effectively captures fine fractional transparency details—like the strands of hair you see in the images—which are critical for a professional look.

2. Solves the 'Flicker' Problem with Temporal Guidance A common failure in video processing is temporal inconsistency, where the mask jitters from frame to frame. This paper specifically integrates Temporal Guidance. By using information from previous frames to guide the current one, it ensures the extracted foreground remains smooth and flicker-free, which is essential for video continuity.

## 4~5

Let's examine the architecture. While this model utilizes blocks from MobileNetV3 as the backbone for the Encoder. Here are the three key differences:

1. The Shift to a Recurrent Architecture (ConvGRU)
   Standard MobileNetV3 is a 'feed-forward' network, it processes one image and forgets it. To handle video, this architecture introduces a Recurrent Decoder. You will notice ConvGRU (Convolutional Gated Recurrent Unit) blocks inserted in both the Bottleneck and the Decoder stages. This allows the model to pass memory from previous frames to the current one, ensuring temporal consistency.

2. Enhanced Context with LR-ASPP In the Bottleneck Block, instead of a simple valid padding or pooling layer, this model employs an LR-ASPP module. This allows the network to capture image context at multiple scales, which is crucial for distinguishing foreground from background in complex scenes.

> LR-ASPP 全称为 Lite Reduced Atrous Spatial Pyramid Pooling（轻量级缩减空洞空间金字塔池化）。
> 它是 MobileNetV3 论文中提出的一种专为移动端设备设计的轻量级分割解码器（Decoder）模块。简单来说，它是经典 ASPP（DeepLab 系列中使用的空洞空间金字塔池化）的“瘦身版”，旨在保持高性能的同时极大降低计算量。
> 位置：它紧接在 Encoder（编码器）输出之后，但在进入 ConvGRU（循环单元）之前。
> 目的：Encoder 输出的特征图虽然分辨率降低了，但包含了丰富的语义信息。LR-ASPP 在这里负责**“总结”这些特征的全局上下文**（例如，快速判断出画面主体大概在哪个区域），并将精炼后的特征传递给 ConvGRU 进行时序处理。

3. High-Resolution Refinement (DGF) Finally, a standard MobileNet typically outputs a lower-resolution classification or map. This architecture includes a dedicated Upsampler and a DGF (Deep Guided Filter) module at the very end. This specifically combines the high-resolution input (Image HR) with the decoder's features to restore fine details like hair strands, which would otherwise be lost.

## 6

Now, let's look at the core of our temporal processing: the ConvGRU inside the Bottleneck Block.

Why did we choose ConvGRU?

The reason is simple: Efficiency. As noted on the slide, ConvGRU is much more parameter-efficient than the traditional ConvLSTM because it uses fewer gates. This makes the model lighter and faster to run.

Functionally, these equations represent a 'gating mechanism.' This allows the network to automatically learn what information to keep and what to forget as it watches a continuous stream of video. This memory is exactly what prevents the flickering issues we discussed earlier.

## 7

Finally, we arrive at the Output Block, which generates our final results.

You might notice a key design choice here: unlike the previous blocks, we intentionally do not use ConvGRU at this stage. We found that at this high resolution scale, using ConvGRU is too computationally expensive and doesn't actually improve the results significantly.

Instead, we stick to regular convolutions to refine the details. The block combines the upsampled features with the original input image and projects them into three distinct outputs:

A 1-channel Alpha prediction (the transparency map),

A 3-channel Foreground prediction (the actual colors of the subject),

And a Segmentation prediction, which helps guide the training objective.

## 8

This brings us to the final piece of the puzzle: the Guided Upsampling Refiner.

We use this mathematical model to ensure our high-resolution output is crisp and detailed. The core idea relies on a simple but powerful assumption: within any small local window of pixels, the output image has a linear relationship with the original high-resolution image.

Instead of using a complex neural network to guess every pixel at full resolution, we mathematically calculate the optimal linear coefficients—$a_k$ and $b_k$.

By minimizing the cost function you see in the middle, we derive the closed-form solutions shown at the bottom. This essentially allows us to 'borrow' the fine structural details (like edges and textures) from the original high-res image and apply them to our alpha matte, mathematically guaranteeing a sharp result.

## 12

- Client Side: The browser captures video streams from the user's cameras, draws them onto a Canvas, and transmits the frames as Base64-encoded strings via a WebSocket connection.
- Server Side: A Python backend receives the data, manages active connections, and feeds the frames into a Deep Learning Pipeline.
- Processing: The system uses a Matting Network to separate the foreground (fgr) from the alpha matte (pha), generating a processed composite image in real-time.
- Broadcast: The processed frame is JPEG-encoded and broadcasted back to the clients, where the browser dynamically updates the display.

## 13

So, here’s a quick demo of the app. This is a screen recording from my phone. The first clip is just me sitting at home, caught on my selfie cam.

And here’s the view from the back lens. I'm playing a video on my computer. I filmed this video while riding the GoldenPass Express in Switzerland a few days ago.

Now, ideally, I should be outside somewhere scenic to show this off, but Dublin is currently freezing. I don't want to go outside, so we’re doing it this way instead!

And there you go—that’s the final composite.

Picture this: you’re at some stunning spot and want to grab a selfie. This app basically acts as your personal director for framing and composition. By using both cameras at once, it lets you stay in focus while perfectly capturing the scenery behind you at the same time.

## 14

In addition to real-time camera feed, our app also supports using video files as backgrounds. I filmed this fireworks show at Disneyland Paris two weeks ago, and now I can head back to the scene—this time with some fresh popcorn!
