# ColabFold

## ColabFold Abstract

* Paper 1

ColabFold offers accelerated prediction of protein structures and complexes by combining the fast homology search of MMseqs2 with AlphaFold2 or RoseTTAFold. ColabFold’s 40−60-fold faster search and optimized model utilization enables prediction of close to 1,000 structures per day on a server with one graphics processing unit. Coupled with Google Colaboratory, ColabFold becomes a free and accessible platform for protein folding. ColabFold is open-source software available at https://github.com/sokrypton/ColabFold and its novel environmental databases are available at https://colabfold.mmseqs.com.

> ColabFold 将 MMseqs2 的快速同源搜索与 AlphaFold2 或 RoseTTAFold 相结合，实现蛋白质结构及复合物的加速预测。ColabFold 的搜索速度提升 40‑60 倍，加上经过优化的模型调用方式，在配备单块图形处理器的服务器上，每日可完成近 1000 个结构的预测。搭配 Google Colaboratory 使用时，ColabFold 就成为一个免费易用的蛋白质折叠平台。ColabFold 是开源软件，可访问
https://github.com/sokrypton/ColabFold 获取，其全新环境数据库可访问
https://colabfold.mmseqs.com 获取


* Paper 2
  
Since its public release in 2021, AlphaFold2 (AF2) has made investigating biological questions, by using predicted protein structures of single monomers or full complexes, a common practice. ColabFold-AF2 is an open-source Jupyter Notebook inside Google Colaboratory and a command-line tool that makes it easy to use AF2 while exposing its advanced options. ColabFold-AF2 shortens turnaround times of experiments because of its optimized usage of AF2’s models. In this protocol, we guide the reader through ColabFold best practices by using three scenarios: (i) monomer prediction, (ii) complex prediction and (iii) conformation sampling. The first two scenarios cover classic static structure prediction and are demonstrated on the human glycosylphosphatidylinositol transamidase protein. The third scenario demonstrates an alternative use case of the AF2 models by predicting two conformations of the human alanine serine transporter 2. Users can run the protocol without computational expertise via Google Colaboratory or in a command-line environment for advanced users. Using Google Colaboratory, it takes <2 h to run each procedure. The data and code for this protocol are available at https://protocol.colabfold.com.

> 自2021年公开发布以来，AlphaFold2（AF2）使利用单体或完整复合物的蛋白质预测结构开展生物学问题研究成为常规手段。ColabFold‑AF2是搭载于Google Colaboratory的开源Jupyter Notebook，同时也是一款命令行工具，可便捷使用AF2并开放其高级选项。ColabFold‑AF2通过优化调用AF2模型，缩短了实验周转时间。在本实验方案中，我们结合三种场景向读者介绍ColabFold的最佳实践：(i)单体预测、(ii)复合物预测以及(iii)构象采样。前两种场景为经典的静态结构预测，以人糖基磷脂酰肌醇转酰胺酶蛋白作为演示对象；第三种场景通过预测人丙氨酸‑丝氨酸转运蛋白2的两种构象，展示AF2模型的另一类应用场景。无计算专业背景的用户可通过Google Colaboratory运行本方案，高级用户则可在命令行环境下使用。借助Google Colaboratory，每项流程运行时长小于2小时。本实验方案的相关数据与代码可访问
https://protocol.colabfold.com 获取。

## ColabFold Model/Method

<placeholder: architecture, input/output, training data, post-processing.>

## Reimplementations / Resources

- LocalColabFold: https://github.com/YoshitakaMo/localcolabfold
