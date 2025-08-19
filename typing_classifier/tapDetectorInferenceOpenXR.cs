using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using UnityEngine.XR.Hands;
using UnityEngine.UI; // <-- NEW: Add this line to use UI elements

/// <summary>
/// Detects a "tap" gesture using the OpenXR hand tracking data and a Sentis model.
/// This script is a direct replacement for the OVR-based TapDetector.
/// </summary>
public class TapDetectorOpenXR : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    [Tooltip("The minimum model confidence needed to register a tap.")]
    [Range(0, 1)] public float detectionThreshold = 0.7f;

    [Header("Hand Tracking")]
    [Tooltip("Which hand should be used for tap detection?")]
    public Handedness trackedHand = Handedness.Right;

    [Header("Feedback UI")]
    [Tooltip("A UI element (e.g., an Image) to show when a tap is detected.")]
    public GameObject tapIndicator;

    [Header("Feedback UI")]
    [Tooltip("A UI element (e.g., an Image) to show when a tap is detected.")]
    public Slider confidenceSlider;

    // --- Sentis Model ---
    private Worker worker;
    private Model runtimeModel;

    // --- Data Management ---
    private const int WINDOW_SIZE = 100;
    private const int NUM_FEATURES = 6; // Z-Position, Z-Velocity, Z-Acceleration

    private readonly Queue<float[]> featureWindow = new Queue<float[]>();

    // --- State for Feature Calculation ---
    private bool isInitialized = false;
    private Vector3 lastPosition;
    private Vector3 lastVelocity;

    // --- OpenXR Hand Subsystem ---
    private XRHandSubsystem handSubsystem;
    private XRHand hand;

    // --- Scaler values for feature normalization ---
    private readonly float[] scalerMean = new float[6] { 0.86914025f, -0.00037746f, -0.03515386f, 0.32091813f, 0.00010341f, 0.04051685f };
    private readonly float[] scalerScale = new float[6] { 0.04979967f, 0.17790756f, 44.22439719f, 0.02788436f, 0.14820831f, 36.96062780f };

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute); // or BackendType.CPU

        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            featureWindow.Enqueue(new float[NUM_FEATURES]);
        }
        
        InitializeHandSubsystem();

        if (tapIndicator != null)
        {
            tapIndicator.SetActive(false);
        }
    }

    void InitializeHandSubsystem()
    {
        var subsystems = new List<XRHandSubsystem>();
        SubsystemManager.GetSubsystems(subsystems);

        if (subsystems.Count == 0)
        {
            Debug.LogError("No XR Hand Subsystem found. Please ensure OpenXR and the XR Hands package are correctly configured in your project.");
            return;
        }

        handSubsystem = subsystems[0];
        Debug.Log("XR Hand Subsystem initialized successfully.");
    }

    void OnDisable()
    {
        worker?.Dispose();
    }

    void Update()
    {
        if (handSubsystem == null) return;

        hand = (trackedHand == Handedness.Left) ? handSubsystem.leftHand : handSubsystem.rightHand;

        if (!hand.isTracked)
        {
            isInitialized = false;
            return;
        }
        
        XRHandJoint indexTipJoint = hand.GetJoint(XRHandJointID.IndexTip);

        if (indexTipJoint.TryGetPose(out Pose pose))
        {
            // Convert world position to local position
            //Vector3 localPos = Camera.main.transform.InverseTransformPoint(pose.position);
            ProcessHandData(pose.position);
        }
        else
        {
            Debug.LogError("Failed to get index tip joint pose");
        }
    }
    
    private void ProcessHandData(Vector3 currentPosition)
    {
        

        if (!isInitialized)
        {
            lastPosition = currentPosition;
            lastVelocity = Vector3.zero;
            isInitialized = true;
            return;
        }

        float dt = Time.deltaTime;
        if (dt <= 0f) return;

        Vector3 currentVelocity = (currentPosition - lastPosition) / dt;
        Vector3 currentAcceleration = (currentVelocity - lastVelocity) / dt;

        var currentFeatures = new float[NUM_FEATURES]
        {
            currentPosition.y,
            currentVelocity.y,
            currentAcceleration.y,
            currentPosition.z,
            currentVelocity.z,
            currentAcceleration.z
        };

        var scaledFeatures = new float[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            // The formula is: (value - mean) / scale
            scaledFeatures[i] = (currentFeatures[i] - scalerMean[i]) / scalerScale[i];
        }

        featureWindow.Dequeue();
        featureWindow.Enqueue(scaledFeatures);

        float[] flatInput = featureWindow.SelectMany(f => f).ToArray();
        using var input = new Tensor<float>(new TensorShape(1, WINDOW_SIZE, NUM_FEATURES), flatInput);

        worker.Schedule(input);
        
        var output = worker.PeekOutput() as Tensor<float>;
        
        var arr = output.DownloadToArray();
        float prediction = arr[0];

        // --- NEW: Update the confidence slider every frame ---
        if (confidenceSlider != null)
        {
            confidenceSlider.value = prediction;
        }

        if (prediction > detectionThreshold)
        {
            Debug.Log($"TAP DETECTED! Confidence: {prediction:P0}");
            if (tapIndicator != null && !tapIndicator.activeInHierarchy)
            {
                StartCoroutine(ShowTapIndicator());
            }
        }

        lastPosition = currentPosition;
        lastVelocity = currentVelocity;
    }
    
    private IEnumerator ShowTapIndicator()
    {
        tapIndicator.SetActive(true);
        yield return new WaitForSeconds(0.05f);
        tapIndicator.SetActive(false);
    }
}
