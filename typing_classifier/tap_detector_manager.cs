using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;

public class TapDetector : MonoBehaviour
{
    [Header("Model Settings")]
    [Tooltip("The ONNX model asset file.")]
    public ModelAsset modelAsset;
    [Tooltip("The probability threshold for detecting a tap.")]
    [Range(0, 1)]
    public float detectionThreshold = 0.7f;

    [Header("Hand Tracking References")]
    [Tooltip("Reference to the OVRHand for the right hand.")]
    public OVRHand rightHand;
    [Tooltip("Reference to the OVRSkeleton for the right hand.")]
    public OVRSkeleton rightHandSkeleton;

    // Sentis model execution engine
    private IWorker engine;
    // The size of the input window our model expects (e.g., 100 frames)
    private const int windowSize = 100;
    // The number of features per frame (e.g., pos_z, vel_z, accel_z)
    private const int numFeatures = 3;

    // A queue to hold the sliding window of recent feature data
    private Queue<float[]> featureWindow = new Queue<float[]>();

    // Variables for calculating velocity and acceleration
    private Vector3 lastPosition;
    private Vector3 lastVelocity;
    private float lastTimestamp;
    private bool isInitialized = false;

    void Start()
    {
        // 1. Load the model and create the inference engine
        Model model = ModelLoader.Load(modelAsset);
        engine = WorkerFactory.CreateWorker(BackendType.GPU, model);

        // Initialize the feature window with zero-filled data
        for (int i = 0; i < windowSize; i++)
        {
            featureWindow.Enqueue(new float[numFeatures]);
        }
    }

    void OnDestroy()
    {
        // Clean up the Sentis engine when the object is destroyed
        engine?.Dispose();
    }

    void Update()
    {
        // Ensure hand tracking is active and initialized
        if (!rightHand.IsTracked)
        {
            isInitialized = false; // Reset if tracking is lost
            return;
        }

        // 2. Get Live Hand Data
        // We'll use the Right Index Fingertip (Distal joint)
        OVRBone indexTipBone = rightHandSkeleton.Bones
            .FirstOrDefault(b => b.Id == OVRSkeleton.BoneId.Hand_IndexTip);

        if (indexTipBone == null) return;

        Vector3 currentPosition = indexTipBone.Transform.position;
        float currentTimestamp = Time.time;

        // Initialize on the first valid frame
        if (!isInitialized)
        {
            lastPosition = currentPosition;
            lastVelocity = Vector3.zero;
            lastTimestamp = currentTimestamp;
            isInitialized = true;
            return;
        }

        // 3. Calculate Live Features
        float deltaTime = currentTimestamp - lastTimestamp;
        if (deltaTime <= 0) return; // Avoid division by zero

        Vector3 currentVelocity = (currentPosition - lastPosition) / deltaTime;
        Vector3 currentAcceleration = (currentVelocity - lastVelocity) / deltaTime;

        // Create the feature array for the current frame
        float[] currentFeatures = new float[numFeatures]
        {
            currentPosition.z,
            currentVelocity.z,

            currentAcceleration.z
        };

        // 4. Update the Sliding Window
        // Add the new frame's data and remove the oldest
        featureWindow.Dequeue();
        featureWindow.Enqueue(currentFeatures);

        // 5. Run Inference
        // Convert the queue of arrays into a single flat array
        float[] flatInput = featureWindow.SelectMany(f => f).ToArray();
        // Create a tensor with the correct 3D shape: (1, windowSize, numFeatures)
        using Tensor inputTensor = new TensorFloat(new TensorShape(1, windowSize, numFeatures), flatInput);

        // Execute the model
        engine.Execute(inputTensor);

        // Get the output tensor
        TensorFloat outputTensor = engine.PeekOutput() as TensorFloat;
        outputTensor.MakeReadable();
        float prediction = outputTensor[0];

        // 6. Make a Decision
        if (prediction > detectionThreshold)
        {
            // We detected a tap!
            Debug.Log($"TAP DETECTED! Confidence: {prediction:P0}");
            // You can add a sound effect or visual feedback here
        }

        // Update state for the next frame
        lastPosition = currentPosition;
        lastVelocity = currentVelocity;
        lastTimestamp = currentTimestamp;
    }
}