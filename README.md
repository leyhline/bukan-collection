# Detection of Differences between Printed Pages and its Application on Bukan

This is a project I worked on during my internship at the National Institute of Informatics in Tokyo, Japan. Essentially, it is about the application of classical Computer Vision on historic japanese literature, thus classifying as *Digital Humanitites*.

Bukan are books from Japan's Edo Period (1603-1868), listing people of influence together with crests, family trees etc. These books were bestsellers, printed using woodblocks. As a result, there is a large number of prints and editions, hiding away potential useful information for the humanities scholar. To lessen the burden on the human researcher a computer might help in comparing the pages, showing recommendations and visualizations.

By utilizing proved techniques from Computer Science and Computer Vision—most notably [Feature Detection](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)) and [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus)—a database is populated with matching pages between different prints. This approach has an accuracy of above 95% when looking for the same page in a different print of a book. Furthermore, this can be used for creating a nice-looking overlay of a page-pair, thus resulting in a useful visualization to quickly discern the differences.

For more details, just have a look at my [Internship Report](InternshipReport.pdf) where I also included a few graphics and examples.