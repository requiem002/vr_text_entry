using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;
using System.Collections;


//worker = SentisWorkerFactory.CreateWorker(SentisBackendType.GPUCompute, runtimeModel);


public class TapDetector : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    [Range(0, 1)] public float detectionThreshold = 0.7f;

    [Header("Hand Tracking References")]
    public OVRHand rightHand;
    public OVRSkeleton rightHandSkeleton;

    private Worker worker;                       // <-- IWorker
    private Model runtimeModel;

    private const int windowSize = 100;
    private const int numFeatures = 3;

    [Header("Feedback UI")]
    [Tooltip("A UI element (e.g., an Image) to show when a tap is detected.")]
    public GameObject tapIndicator; // <-- NEW: Reference to our UI element

    private readonly Queue<float[]> featureWindow = new Queue<float[]>();
    private Vector3 lastPosition, lastVelocity;
    private float lastTimestamp;
    private bool isInitialized = false;

    void Start()
    {
        // Load ONNX and create runtime model (no Functional.Compile needed unless you modify the graph)
        runtimeModel = ModelLoader.Load(modelAsset);

        // Create an engine (GPUCompute backend; falls back to CPU if unavailable)
        //worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel); // or BackendType.CPU
        // :contentReference[oaicite:1]{index=1}

        worker = new Worker(runtimeModel, BackendType.GPUCompute); // or BackendType.CPU

        // Make sure the indicator is off at the start
        if (tapIndicator != null)
        {
            tapIndicator.SetActive(false);
        }

        for (int i = 0; i < windowSize; i++)
            featureWindow.Enqueue(new float[numFeatures]);
    }

    void OnDisable()
    {
        worker?.Dispose(); // :contentReference[oaicite:2]{index=2}
    }

    void Update()
    {
        if (!rightHand || !rightHand.IsTracked || rightHandSkeleton == null)
        { isInitialized = false; return; }

        var indexTipBone = rightHandSkeleton.Bones
            .FirstOrDefault(b => b.Id == OVRSkeleton.BoneId.Hand_IndexTip);

        if (indexTipBone == null)
        {
            Debug.Log("Could not find Index Tip Bone!");
            return;
        }

        Vector3 currentPosition = indexTipBone.Transform.position;
        float currentTimestamp = Time.time;

        if (!isInitialized)
        {
            lastPosition = currentPosition;
            lastVelocity = Vector3.zero;
            lastTimestamp = currentTimestamp;
            isInitialized = true;
            return;
        }

        float dt = currentTimestamp - lastTimestamp;
        if (dt <= 0f) return;

        Vector3 currentVelocity = (currentPosition - lastPosition) / dt;
        Vector3 currentAcceleration = (currentVelocity - lastVelocity) / dt;

        var currentFeatures = new float[numFeatures]
        {
            currentPosition.z,
            currentVelocity.z,
            currentAcceleration.z
        };

        featureWindow.Dequeue();
        featureWindow.Enqueue(currentFeatures);

        float[] flatInput = featureWindow.SelectMany(f => f).ToArray();
        
        using var input = new Tensor<float>(new TensorShape(1, windowSize, numFeatures), flatInput);

        //using var inputTensor = new TensorFloat(new TensorShape(1, windowSize, numFeatures), flatInput);

        worker.Schedule(input);

        // If your model has one output, parameterless PeekOutput() is fine.
        //var outputTensor = worker.PeekOutput() as TensorFloat;

        var output = worker.PeekOutput() as Tensor<float>;
        //outputTensor.CompleteOperationsAndDownload(); // make it CPU-readable :contentReference[oaicite:4]{index=4}
        
        
        var arr = output.DownloadToArray();
        //float prediction = outputTensor[0];
        float prediction = arr[0];


        if (prediction > detectionThreshold)
            Debug.Log($"TAP DETECTED! Confidence: {prediction:P0}");

            // --- NEW: Trigger the visual feedback ---
            if (tapIndicator != null && !tapIndicator.activeInHierarchy)
            {
                StartCoroutine(ShowTapIndicator());
            }

        

        lastPosition = currentPosition;
        lastVelocity = currentVelocity;
        lastTimestamp = currentTimestamp;
    }

    // --- NEW: Coroutine to show the indicator for a short time ---
    private IEnumerator ShowTapIndicator()
    {
        Debug.Log("Showing tap indicator");
        tapIndicator.SetActive(true);
        // Wait for 0.1 seconds
        yield return new WaitForSeconds(0.05f);
        tapIndicator.SetActive(false);
    }
}