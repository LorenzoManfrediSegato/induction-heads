## Zoom In thoughts / notes

### The three claims, in my own words

1. Features: Features are the most fundamental unit in neural networks; that doesn't necessarily mean the most elementary or primitive, just the smallest one that has any meaning. A good analogy are words. Letters are more primitive/elementary, but in most contexts a letter on its own doesn't tell much of a story.

2. Circuits: Circuits exist! They can be seen as the progession of features with weights along multiple layers. They are essentially subgraphs across different neurons/channels in the network. Interestingly, these circuits show that the features from which we see progression are related: Early features that show edge detection may later show curve detection, which may later become circle-detection features and so forth.

3. Universality: This is cool and interesting. Similar features / circuits form across different models and tasks. The same model architecture trained on different data annecdotally come up with the same features emerge. Similar results seem to follow from different architecture with same data. The paper doesn't provide much evidence and the authors are open about that but the opportunities are interesting, if universality holds: once one understands a model, one also learns about others - simiar to studying biological organisms. 


### What convinced me

For the first couple of claims, the authors presented a fair amount of evidence, particularly related to curve detection. The 7 arguments presented on the first claim were particularly convincing, especially the first three. High-low frequency detectors were interesting and in particular the pose-invariant dog head detector. To find such an (at least to me) unintuitive/unhuman way to 'break down' the understanding from the data the model is trying to learn was convincing. The fact that it was unintuitive had nothing to do with the fact that it was convincing - it's more that feature visualization along with the dataset examples they provided (like they mention in the paper) really sold me on it. Honestly, I was kind of sold from the start.

The argument they provide describing the establishment of the causal link and the testing of the neuron's use in practice, 'reacting' how we expected to was great.

### Weights

I also thought the *readability* of these feature detectors was interesting. Due to being a convolutional neural network, layers are connected and we can see how. It was awesome to see the car example - how the earlier-layer features of windows, bodies, and wheels output a car in a later layer. The connections via the weights was cool, they specify how parts are arranged relatively to each other.

### Polysemantics

These are cool. Interesting to see how neurons that describe certain features don't behave how we may expect them to. I also like their hypothesis on why polysemantic neurons occur -to save space. I thought the implications of this are cool: Does network (or circuit/feature?) 'know' how many more layers there are left? Polysemantics kind of 'gamble' on making a neuron good at detecting cat legs and the fronts of cars, because it relies on the fact that future layers will be able to discern these more accurately. What if there are no more layers? Are these neurons just not good enough then? Is the network just not big enough and therefore unreliable/bad? Is this just bad design? Or is it 'calculating' based on how 'refined' the current feature is? Maybe not so early as to be an edge detector, but maybe not so late as to be a Siberian Husky eyelid detector.

Bummer that they kind of get in the way of circuits though, makes things less clean cut and clear - but I do find this fascinating. I want to understand more about these - how they come about, why, and their impact? Is there a better way here? Or is gradient descent just this goated?

### Ex Nihilo

This was a cool paper to read to understand things from scratch. How fundamental units, features, build up and come about to be rather sophisticated detectors. Didn't really get to see the whole picture - I suppose that is a task of Herculean proportions, even for today's standards but it was interesting. I really enjoyed the how feature detectors came about, seemingly arcanely through gradient descent and kernels convoluting across channels.

### Open questions

1. Polysemanticity: Does the network 'know'? Is it taking a gamble? Is this somehow related to how/why features and feature detectors show up?
