
# Децентрализованное обучение глубоких нейронных сетей в распределенной вычислительной среде

## План

1. Конец сентября - 1й чекпойнт (установочный): создание Github, встреча с руководителем, примерный план работы, ознакомление с темой
2. 20-e числа октября - 2й чекпойнт (разведочный анализ или аналог): выбор датасетов и архитектур глубоких нейронных сетей для решения задач компьютерного зрения, анализ возможных подходов к параллелизму данных, реализация синхронного подхода на выбранном датасете. Получение предварительных результатов
3. Конец ноября - 3й чекпойнт (линейные ML-модели или аналог): реализация асинхронного параллелизма данных, сравнение с синхронным. Попробовать еще один или несколько датасетов, другие способы распределения данных по локальным датасетам.
4. Середина января - 4й чекпойнт (нелинейные ML-модели или аналог): сравнительный анализ методов синхронного и асинхронного подходов по метрикам качества.
5. Конец января - промежуточная защита
6. Середина февраля - 5й чекпойнт (сервис): анализ подходов к параллелизму модели. Реализация одного из подходов на выбор, например наивного параллелизма или пайплайн параллелизма.
7. Конец марта-начало апреля - 6й чекпойнт (внедрение DL или аналог): использование полученных результатов для решения прикладной задачи. Исследование гибридных подходов.
8. Середина мая - 7й чекпойнт (улучшение моделей и сервиса): улучшения метрик качества, исправление ошибок и доработка реализации. Получение финальных результатов.
9. Июнь - защита

## Участники проекта

- Александров Максим
- Троицкий Максим
- Баранов Егор
- Костина Полина

Куратор проекта - Курочкин Илья

## Введение

В настоящее время распределенное обучение стремительно набирает популярность вследствие роста масштабов современных нейронных сетей. Так, модель “DeepSeek-V3.1-Base”  содержит около 685 миллиардов параметров, тогда как ее предыдущая версия “DeepSeek V2”  включала лишь порядка 236 миллиардов. Объем параметров увеличился почти втрое, что существенно усложняет процесс обучения и развёртывания, особенно в случае использования не квантованных моделей. При этом увеличивается не только количество параметров моделей, но и объем обучающих наборов данных. Эта тенденция приводит к необходимости масштабирования вычислений и оптимизации систем ввода-вывода, что делает распределенное обучение одним из ключевых направлений развития области машинного обучения.
Распределенное обучение может помочь снизить необходимое количество ресурсов для обучения моделей и позволить сократить издержки на дорогие GPU. Существует несколько подходов к распределенному обучению глубоких нейронных сетей [2]. Основные два — это параллелизм данных и параллелизм модели. Параллелизм данных подразумевает: (1) разделение датасета между несколькими вычислительными узлами, (2) запуск модели на каждой из них с обучением на её части набора данных, (3) синхронизацию весов модели после каждой итерации обучения. При параллелизме модели слои разделяются между несколькими вычислительными узлами и передают друг другу активации по сети или в рамках одного узла с несколькими GPU, например, с помощью MPI [45] или NCCL [44]. В общем случае, когда количество активаций превышает количество параметров модели, параллелизм по данным создаёт меньшую нагрузку на сеть, поскольку объём передаваемых данных будет меньше [10].
Существует несколько алгоритмов коммуникации, используемых для синхронизации весов модели после обработки обучающих данных в подходе с параллелизмом данным. Их можно условно разделить на централизованные, чаще всего реализуемые через параметрический сервер [7] [4], и децентрализованные. Наиболее распространённый подход к децентрализованной синхронизации — это использование "tree reduction" [10]. Другие способы передачи параметров включают, например, топологию с разделённой линейностью [9] или ring-AllReduce, используемую в фреймворке Horovod [13]. Методы синхронизации также можно разделить на три основные категории [2]:

1. Синхронный параллелизм данных – каждая копия модели обучается на своём подмножестве данных, после каждой итерации происходит обмен и усреднение градиентов.
2. Асинхронный параллелизм данных – узлы обновляют параметры независимо, что снижает задержки, но может привести к расхождению и устареванию градиентов.
3. Локально-синхронный (гибридный) параллелизм – смешанный подход, при котором вычисления выполняются одновременно на разных уровнях.

Существует также несколько подходов к разделению датасета между рабочими машинами. Различают локальное, глобальное и частично-локальное перемешивание [12]. При локальном перемешивании каждый рабочий узел получает подмножество набора данных перед обучением и затем перемешивает это подмножество локально на каждой итерации. При глобальном перемешивании весь набор данных собирается с использованием распределенной системы хранения/файловой системы, такой как Lustre FS [43], а затем случайным образом делится на M частей, где M — количество рабочих узлов. Глобальное перемешивание требует системы хранения, достаточно большой для размещения всего набора данных, и высокого объёма коммуникации, что создает значительного нагрузку на сеть. При частично-локальном перемешивании данных датасет также сначала делится на M частей. Также выбирается некоторая константа Q ϵ [0, 1]  По мере увеличения числа рабочих узлов и уменьшения размеров частей данных локальное перемешивание начинает работать хуже. Однако в большинстве задач локальное перемешивание достигает такой же top-1 accuracy, как и глобальное перемешивание. Частично-локальное перемешивание позволяет достичь точности, сравнимой с глобальным перемешиванием, при этом значительно снижая объем коммуникации. Для большого вычислительного кластера групповую нормализацию [40] можно рассматривать как альтернативу нормализации по батчам, поскольку она, предположительно, лучше работает с меньшими наборами данных, однако требуются дополнительные эксперименты, чтобы подтвердить это.
Техники, такие как PCA [36], [37], "weight pruning" [38] и "gradient clipping", могут использоваться для снижения нагрузки на сеть при распределенном обучении.

В  статье Sergey Zagoruyko, Nikos Komodakis “Wide Residual Networks” (2017)  было показано, что увеличение ширины слоев модели может быть столь же эффективным, как и увеличение глубины. Например, широкие остаточные сети (Wide ResNets) демонстрируют ту же точность, что и их “тонкие” аналоги с тысячами слоёв, при этом требуют примерно в 50 раз меньшей глубины и обучаются более чем в два раза быстрее. Этот результат важен в контексте нашего исследования, так как баланс между шириной и глубиной модели напрямую влияет на эффективность распределенного обучения и использование вычислительных ресурсов. Дополнительно, в статье Tasnim Shahriar “Comparative Analysis of Lightweight Deep Learning Models for Memory-Constrained Devices” (2025) было проведено сравнение моделей MobileNetV3, ResNet18, SqueezeNet, EfficientNetV2 и ShuffleNetV2 на трёх наборах данных различной сложности. Авторы оценивали точность, время инференса, количество операций (FLOPs) и размер модели. Результаты показали, что модель EfficientNetV2 обеспечивает наивысшую точность на всех наборах данных, однако её большие размеры и время инференса ограничивают применимость в условиях ограниченных ресурсов. В то же время MobileNetV3 продемонстрировала наилучший баланс между скоростью, точностью и компактностью, что делает её оптимальной для распределённых или периферийных систем, где критичны задержки и объём памяти. В статье Zhuang Liu, Hanzi Mao, Chao-Yuan Wu et al. “A ConvNet for the 2020s” (2022) авторы модернизировали сверточные сети, черпая идеи из Swim Transformers, таким образом удалось создать семейство ConvNeXt –  полностью сверточные модели, которые на задачах классификации (например, ImageNet-1K) и задачах детекции/семантики (COCO, ADE20K) показали результаты, сопоставимые или превосходящие современные трансформеры. Дополнительным преимуществом этой архитектуры является ее масштабируемость, так как для нашего исследования придется увеличивать количество узлов и объем данных, то важно, чтобы архитектура выдерживала масштабирование. ConvNeXt как раз демонстрирует такое поведение.

## Литература
1. Ben-Nun, T.; Hoefler, T. Demystifying Parallel and Distributed Deep Learning: An In-depth Concurrency Analysis. ACM Computing Surveys. 2019, 52.
2. Langer, M.; He, Z.; Rahayu W.; Xue, Y. Distributed Training of Deep Learning Models: A Taxonomic Perspective. IEEE Transactions on Parallel and Distributed Systems. 2020, 31, 12, 2802–2818.
3. Verbraeken, Joost; Wolting, Matthijs; Katzy, Jonathan; Kloppenburg, Jeroen; Verbelen, Tim; Rellermeyer, Jan S. (2020). “A Survey on Distributed Machine Learning.” ACM Computing Surveys, Vol. 53, No. 2, pp. 1–33. Association for Computing Machinery (ACM). DOI: 10.1145/3377454
4. Mu Li, David G. Andersen, Jun Woo Park, Alexander J. Smola, Amr Ahmed,Vanja Josifovski, James Long, Eugene J. Shekita, Bor-Yiing Su. Scaling Distributed Machine Learning with the Parameter Server. https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf 
5. https://www.cyberforum.ru/blogs/2409963/10268.html 
6. Back-up workers https://hal.science/hal-03364773/document
7. S. Jiang, S. Qin, A. Bhattacharjee, and S. Narayanan, "Gandiva: Accelerating Deep Learning with High-Bandwidth Inter-Chip Communication," in Proc. 14th USENIX Symp. Operating Systems Design and Implementation (OSDI 20), Virtual Event, USA, Nov. 2020, pp. 689-705. Available: https://www.usenix.org/system/files/osdi20-jiang.pdf
8. Liang Luo, Jacob Nelson, Luis Ceze, Amar Phanishayee, and Arvind Krishnamurthy. 2018. Parameter Hub: a Rack-Scale Parameter Server for Distributed Deep Neural Network Training. In Proceedings of the ACM Symposium on Cloud Computing (SoCC '18). Association for Computing Machinery, New York, NY, USA, 41–54. https://doi.org/10.1145/3267809.3267840 
9. Y. Zou, X. Jin, Y. Li, Z. Guo, E. Wang, and B. Xiao, "Mariana: Tencent deep learning platform and its applications," Proc. VLDB Endow., vol. 7, no. 13, pp. 1772–1777, Aug. 2014. [Online]. Available: https://www.vldb.org/pvldb/vol7/p1772-tencent.pdf
10. S. Ioffe, F. Bigham, J. Chhugani, C. Kozyrakis, and D. Patterson, "FireCaffe: near-linear acceleration of deep neural network training on compute clusters," University of California, Berkeley, Tech. Rep. EECS-2015-82, June 2015. Available: https://example.com/firecaffe.pdf
11. J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, A. Senior, P. Tucker, K. Yang, Q. V. Le, et al., "Large Scale Distributed Deep Networks," in *Proc. 25th Int. Conf. Neural Inf. Process. Syst.*, Lake Tahoe, Nevada, 2012, pp. 1223-1231. Available: https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf
12. T. T. Nguyen, F. Trahay, J. Domke, A. Drozd, E. Vatai, J. Liao, M. Wahib, and B. Gerofi, "Why globally re-shuffle? Revisiting data shuffling in large scale deep learning," in Proc. IEEE 29th Int. Conf. High Perform. Comput., Data, Anal., Vis. (HiPC), Hyderabad, India, Dec. 2019, pp. 235–245. [Online]. Available: https://www.researchgate.net/publication/362031936_Why_Globally_Re-shuffle_Revisiting_Data_Shuffling_in_Large_Scale_Deep_Learning
13. A. Sergeev and M. Del Balso, "Horovod: Fast and easy distributed deep learning in TensorFlow," arXiv preprint arXiv:1802.05799, 2018. [Online]. Available: https://arxiv.org/pdf/1802.05799.
14. Y. You, I. Gitman, and B. Ginsburg, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," [Online]. Available: https://arxiv.org/pdf/1706.02677
15. S. Li, M. Shoeybi, T. Goodbody, S. Chintala, and P. Vajaria, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," arXiv preprint arXiv:2006.15704, 2020. [Online]. Available: https://arxiv.org/pdf/2006.15704.
16. P. Moritz, R. Nishihara, S. Wang, A. Tumanov, R. Liaw, M. Liang, M. Stoica, and I. Stoica, "Ray: A Distributed Framework for Emerging AI Applications," in Proc. 18th USENIX Symp. Operating Systems Design and Implementation (OSDI 18), Carlsbad, CA, USA, Oct. 2018, pp. 613-628. Available: https://www.usenix.org/system/files/osdi18-moritz.pdf
17. PyTorch. Implementing a Parameter Server Using Distributed RPC Framework. https://docs.pytorch.org/tutorials/intermediate/rpc_param_server_tutorial.html
18. Aach, M., Inanc, E., Sarma, R. et al. Large scale performance analysis of distributed deep learning frameworks for convolutional neural networks. J Big Data 10, 96 (2023). https://doi.org/10.1186/s40537-023-00765-w 
19. Сравнение фреймворков. https://www.netguru.com/blog/deep-learning-frameworks-comparison
20. Zakariya Ba Alawi. A Comparative Survey of PyTorch vs TensorFlow for Deep Learning: Usability, Performance, and Deployment Trade-offs. https://arxiv.org/html/2508.04035v1 
21. Debugging distributed python applications: https://docs.pytorch.org/docs/main/distributed.html#debugging-torch-distributed-applications
22. DDP, DS, HOROVOD тестирование фреймворков на ResNet + загрузчик данных DALI vs classic pytorch https://gitlab.jsc.fz-juelich.de/CoE-RAISE/FZJ/resnet-benchmarks 
23. Распределенное обучение на PyTorch (очень простое объяснение как реализовать) https://telegra.ph/Raspredelyonnoe-obuchenie-s-PyTorch-na-klastere-dlya-teh-kto-speshit-08-11 
24. https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=distr_landing&utm_medium=intermediate_ddp_tutorial 
25. https://translated.turbopages.org/proxy_u/en-ru.ru.82fe442a-68f8d33d-d373c275-74722d776562/https/www.geeksforgeeks.org/deep-learning/distributed-applications-with-pytorch/#practical-example-distributed-training-of-a-resnet-model 
26. ImageNet https://www.image-net.org/download.php 
27. CIFAR-10 https://huggingface.co/datasets/uoft-cs/cifar10 
28. Обзор датасетов: https://pyimagesearch.com/2023/07/31/best-machine-learning-datasets/ 
29. A. Katharopoulos and F. Fleuret, “Not all samples are created equal: Deep learning with importance sampling,” in ICML, 2018, pp. 2530–2539.
30. D. Cheng, S. Li, H. Zhang, F. Xia, and Y. Zhang, “Why dataset properties bound the scalability of parallel machine learning training algorithms,” TPDS, vol. 32, no. 7, pp. 1702–1712, 2021.
31. Сравнение моделей глубокого обучения для классификации изображений (MobileNetV3, ResNet18, SqueezeNet, EfficientNetV2, ShuffleNetV2) на датасетах CIFAR-10/100 и Tiny ImageNet https://arxiv.org/html/2505.03303v1#abstract
32. S. Zagoruyko and N. Komodakis, “Wide residual networks,” in BMVC, September 2016, pp. 87.1–87.12.
33. Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie “A ConvNet for the 2020s”  https://arxiv.org/abs/2201.03545 
34. S. Chetlur, C. Woolley, P. Vandermersch, J. Cohen, J. Tran, B. Catanzaro, and E. Oh, "cuDNN: Efficient Primitives for Deep Learning," arXiv:1410.0759 [cs.DC], Oct. 2014. Available: https://arxiv.org/pdf/1410.0759
35. N. Strom. Scalable distributed dnn training using commodity gpu cloud computing. In INTERSPEECH, 2015.
36. X. Zhang, J. Zou, X. Ming, K. He, and J. Sun. Efficient and accurate approximations of nonlinear convolutional networks. arXiv:1411.4229, 2014
37. M. Jaderberg, A. Vedaldi, and A. Zisserman. Speeding up convolutional neural networks with low rank expansions. arXiv:1405.3866, 2014.
38. S. Han, H. Mao, and W. J. Dally. A deep neural network compression pipeline: Pruning, quantization, huffman encoding. arXiv:1510.00149, 2015.
39. Q. Meng, W. Chen, Y. Wang, Z. Ma, and T. Liu, “Convergence analysis of distributed stochastic gradient descent with shuffling,” Neurocomputing, vol. 337, pp. 46–57, 2019.
40. Y. Wu and K. He, “Group normalization,” in ECCV, 2018, pp. 3–19.
41. Y. You, I. Gitman, and B. Ginsburg, “Large batch training of convolutional networks,” 2017.
42. H. Mikami, H. Suganuma, Y. Tanaka, Y. Kageyama et al., “Massively distributed sgd: Imagenet/resnet-50 training in a flash,” arXiv preprint arXiv:1811.05233, 2018.
43. P. J. Braam, M. D. Brent, and S. M. Welch, "The Lustre Storage Architecture," arXiv preprint arXiv:1903.01955, 2019. [Online]. Available: https://arxiv.org/pdf/1903.01955.
44. NVIDIA, "NVIDIA NCCL," [Online]. Available: https://developer.nvidia.com/nccl
45. Message Passing Interface Forum. “MPI: A Message-Passing Interface Standard”. Available: https://www.mpi-forum.org/docs/mpi-4.0/mpi40-report.pdf
