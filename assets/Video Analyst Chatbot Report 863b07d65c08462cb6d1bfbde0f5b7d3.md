# Video Analyst Chatbot Report

### 1. Proposed pipeline

![image.png](image.png)

### 2. Results

**Test video:** [https://www.youtube.com/watch?v=Pv0iVoSZzN8&vl](https://www.youtube.com/watch?v=Pv0iVoSZzN8&vl=en)

**Sample scene description:**

![image.png](image%201.png)

**Sample audio transcript:**

![image.png](image%202.png)

**Retrieval with metadata (t_start, t_end of an event):**

- The model au to generate filter base on query

![image.png](image%203.png)

- Retrieved node have metadata followed the filter

![image.png](image%204.png)

**Demo chatbot with 3 questions** (File `demo_chatbot_simple.ipynb`)

![image.png](image%205.png)

**Debugging and tracing** (File `demo_chatbot_logging.ipynb`)

![image.png](image%206.png)

**Compare rag technique used in this pipeline:** File `rag_develop_pipelines.ipynb`

### 3. Further Improvement

**Better video splitting strategy:**

- Time based extraction

![image.png](image%207.png)

- Shot based extraction

![image.png](image%208.png)

**Better audio transcript segment strategy:** 

- Default transcript segmented by Whisper model may lead to bad retrieval results
- Combine word level, segment level to improve keyword search

**Enhance metadata search**

- Currently metadata only contain time start and time end of an event
- Apply more techniques to extract more information from video scenes and store in metadata: person detection, face detection, …