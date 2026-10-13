#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

// ==============================================================================
// 1. SPECTRAL SUPPRESSOR DSP ENGINE
// ==============================================================================
class SpectralSuppressorEngine
{
public:
    static constexpr int fftOrder = 11;                // 2048 samples
    static constexpr int fftSize  = 1 << fftOrder;     // 2048
    static constexpr int numBins  = fftSize / 2 + 1;   // 1025 bins
    static constexpr int overlap  = 4;                 // 75% overlap
    static constexpr int hopSize  = fftSize / overlap; // 512 samples

    SpectralSuppressorEngine()
        : fft(fftOrder),
          window(fftSize + 1, juce::dsp::WindowingFunction<float>::hann, false)
    {
        reset();
    }

    void prepare(double newSampleRate)
    {
        sampleRate = newSampleRate;
        reset();
    }

    void reset()
    {
        inputFifo.fill(0.0f);
        outputFifo.fill(0.0f);
        dryDelayLine.fill(0.0f);
        fftData.fill(0.0f);
        smoothedGain.fill(1.0f);
        visualReductionDB.fill(0.0f);
        fifoIndex = 0;
        dryWritePos = 0;
        hopCounter = 0;
    }

    int getLatencySamples() const { return fftSize; }

    float processSample(float inputSample,
                        const std::array<float, numBins>& sidechainWeightsLinear,
                        float depthAmount,
                        float detailAmount,
                        float attackMs,
                        float releaseMs,
                        bool hardMode,
                        bool deltaMode,
                        float mix)
    {
        dryDelayLine[dryWritePos] = inputSample;
        int dryReadPos = (dryWritePos + 1) % fftSize;
        float delayedDry = dryDelayLine[dryReadPos];
        dryWritePos = (dryWritePos + 1) % fftSize;

        inputFifo[fifoIndex] = inputSample;

        float wetSample = outputFifo[fifoIndex];
        outputFifo[fifoIndex] = 0.0f;

        fifoIndex = (fifoIndex + 1) % fftSize;

        if (++hopCounter >= hopSize)
        {
            hopCounter = 0;
            processSpectralFrame(sidechainWeightsLinear, depthAmount, detailAmount,
                                 attackMs, releaseMs, hardMode);
        }

        float wetNormalized = wetSample / 1.5f;

        if (deltaMode)
            return wetNormalized - delayedDry;

        return delayedDry * (1.0f - mix) + wetNormalized * mix;
    }

    const std::array<float, numBins>& getVisualReductionDB() const { return visualReductionDB; }

private:
    void processSpectralFrame(const std::array<float, numBins>& sidechainWeights,
                              float depth, float detail, float attackMs, float releaseMs, bool hardMode)
    {
        for (int i = 0; i < fftSize; ++i)
            fftData[i] = inputFifo[(fifoIndex + i) % fftSize];

        window.multiplyWithWindowingTable(fftData.data(), fftSize);
        std::fill(fftData.begin() + fftSize, fftData.end(), 0.0f);
        fft.performRealOnlyForwardTransform(fftData.data());

        std::array<float, numBins> magDB;
        for (int b = 0; b < numBins; ++b)
        {
            float re = fftData[2 * b];
            float im = fftData[2 * b + 1];
            float m  = std::sqrt(re * re + im * im) / static_cast<float>(fftSize);
            magDB[b] = juce::Decibels::gainToDecibels(m, -100.0f);
        }

        int baselineRadius = static_cast<int>(juce::jmap(detail, 0.0f, 1.0f, 24.0f, 6.0f));
        std::array<float, numBins> baselineDB;
        for (int b = 0; b < numBins; ++b)
        {
            int startBin = std::max(1, b - baselineRadius);
            int endBin   = std::min(numBins - 1, b + baselineRadius);
            float sum = 0.0f;
            for (int k = startBin; k <= endBin; ++k)
                sum += magDB[k];
            baselineDB[b] = sum / static_cast<float>(endBin - startBin + 1);
        }

        float frameRate = static_cast<float>(sampleRate) / static_cast<float>(hopSize);
        float attCoeff  = std::exp(-1.0f / (std::max(0.5f, attackMs) * 0.001f * frameRate));
        float relCoeff  = std::exp(-1.0f / (std::max(5.0f, releaseMs) * 0.001f * frameRate));

        float maxReductionDB = hardMode ? -24.0f : -14.0f;
        float ratioSlope     = hardMode ? 0.75f  : 0.45f;
        float threshOffsetDB = juce::jmap(depth, 0.0f, 1.0f, 18.0f, -6.0f) - (hardMode ? 3.0f : 0.0f);

        std::array<float, numBins> rawTargetGain;
        for (int b = 0; b < numBins; ++b)
        {
            float weightDB     = juce::Decibels::gainToDecibels(sidechainWeights[b], -60.0f);
            float prominenceDB = (magDB[b] - baselineDB[b]) + weightDB;
            float absoluteGate = (magDB[b] > -75.0f) ? 1.0f : 0.0f;

            float overDB = (prominenceDB - threshOffsetDB) * depth * absoluteGate;
            float reductionDB = (overDB > 0.0f) ? std::max(maxReductionDB, -overDB * ratioSlope) : 0.0f;
            rawTargetGain[b] = juce::Decibels::decibelsToGain(reductionDB);
        }

        int smoothBins = static_cast<int>(juce::jmap(1.0f - detail, 0.0f, 1.0f, 1.0f, 12.0f));
        for (int b = 0; b < numBins; ++b)
        {
            int s = std::max(0, b - smoothBins);
            int e = std::min(numBins - 1, b + smoothBins);
            float gSum = 0.0f;
            for (int k = s; k <= e; ++k)
                gSum += rawTargetGain[k];
            float targetG = gSum / static_cast<float>(e - s + 1);

            if (targetG < smoothedGain[b])
                smoothedGain[b] = attCoeff * smoothedGain[b] + (1.0f - attCoeff) * targetG;
            else
                smoothedGain[b] = relCoeff * smoothedGain[b] + (1.0f - relCoeff) * targetG;

            fftData[2 * b]     *= smoothedGain[b];
            fftData[2 * b + 1] *= smoothedGain[b];
            visualReductionDB[b] = juce::Decibels::gainToDecibels(smoothedGain[b], -30.0f);
        }

        fft.performRealOnlyInverseTransform(fftData.data());
        window.multiplyWithWindowingTable(fftData.data(), fftSize);

        for (int i = 0; i < fftSize; ++i)
            outputFifo[(fifoIndex + i) % fftSize] += fftData[i];
    }

    juce::dsp::FFT fft;
    juce::dsp::WindowingFunction<float> window;
    std::array<float, fftSize> inputFifo {}, outputFifo {}, dryDelayLine {};
    std::array<float, fftSize * 2> fftData {};
    std::array<float, numBins> smoothedGain {}, visualReductionDB {};
    int fifoIndex = 0, dryWritePos = 0, hopCounter = 0;
    double sampleRate = 44100.0;
};

// ==============================================================================
// 2. AUDIO PROCESSOR
// ==============================================================================
class DynamicResonanceSuppressorAudioProcessor : public juce::AudioProcessor
{
public:
    DynamicResonanceSuppressorAudioProcessor()
        : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "Parameters", createParameterLayout())
    {
        sidechainWeightsLinear.fill(1.0f);
        sidechainWeightsDB.fill(0.0f);
    }

    ~DynamicResonanceSuppressorAudioProcessor() override = default;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
    {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"depth", 1}, "Depth (Soothe)", juce::NormalisableRange<float>(0.0f, 10.0f, 0.01f), 4.5f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"detail", 1}, "Detail (Sharpness)", juce::NormalisableRange<float>(0.0f, 10.0f, 0.01f), 5.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"attack", 1}, "Attack", juce::NormalisableRange<float>(0.5f, 100.0f, 0.1f, 0.4f), 10.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"release", 1}, "Release", juce::NormalisableRange<float>(10.0f, 500.0f, 0.1f, 0.4f), 80.0f));
        params.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"hardMode", 1}, "Hard Mode", false));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"lowCutFreq", 1}, "Low Cut Freq", juce::NormalisableRange<float>(20.0f, 2000.0f, 1.0f, 0.3f), 80.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"highCutFreq", 1}, "High Cut Freq", juce::NormalisableRange<float>(2000.0f, 20000.0f, 1.0f, 0.3f), 16000.0f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b1Freq", 1}, "Band 1 Freq", juce::NormalisableRange<float>(40.0f, 18000.0f, 1.0f, 0.3f), 350.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b1Gain", 1}, "Band 1 Depth (dB)", juce::NormalisableRange<float>(-18.0f, 18.0f, 0.1f), 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b1Q", 1}, "Band 1 Q", juce::NormalisableRange<float>(0.3f, 8.0f, 0.05f, 0.5f), 1.2f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b2Freq", 1}, "Band 2 Freq", juce::NormalisableRange<float>(40.0f, 18000.0f, 1.0f, 0.3f), 2500.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b2Gain", 1}, "Band 2 Depth (dB)", juce::NormalisableRange<float>(-18.0f, 18.0f, 0.1f), 4.5f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b2Q", 1}, "Band 2 Q", juce::NormalisableRange<float>(0.3f, 8.0f, 0.05f, 0.5f), 1.4f));

        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b3Freq", 1}, "Band 3 Freq", juce::NormalisableRange<float>(40.0f, 18000.0f, 1.0f, 0.3f), 6500.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b3Gain", 1}, "Band 3 Depth (dB)", juce::NormalisableRange<float>(-18.0f, 18.0f, 0.1f), 3.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"b3Q", 1}, "Band 3 Q", juce::NormalisableRange<float>(0.3f, 8.0f, 0.05f, 0.5f), 1.5f));

        params.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"delta", 1}, "Delta", false));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"mix", 1}, "Mix", juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f), 100.0f));
        params.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"outGain", 1}, "Out Gain", juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f));
        params.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass", 1}, "Bypass", false));

        return { params.begin(), params.end() };
    }

    void prepareToPlay(double sampleRate, int) override
    {
        currentSampleRate = sampleRate;
        for (auto& eng : engines)
            eng.prepare(sampleRate);
        setLatencySamples(engines[0].getLatencySamples());
        updateSidechainWeights();
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override
    {
        if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
         && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
            return false;
        return layouts.getMainOutputChannelSet() == layouts.getMainInputChannelSet();
    }

    void updateSidechainWeights()
    {
        using IIRCoeffs = juce::dsp::IIR::Coefficients<float>;
        float lowCut  = apvts.getRawParameterValue("lowCutFreq")->load();
        float highCut = apvts.getRawParameterValue("highCutFreq")->load();
        float b1F = apvts.getRawParameterValue("b1Freq")->load();
        float b1G = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("b1Gain")->load());
        float b1Q = apvts.getRawParameterValue("b1Q")->load();
        float b2F = apvts.getRawParameterValue("b2Freq")->load();
        float b2G = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("b2Gain")->load());
        float b2Q = apvts.getRawParameterValue("b2Q")->load();
        float b3F = apvts.getRawParameterValue("b3Freq")->load();
        float b3G = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("b3Gain")->load());
        float b3Q = apvts.getRawParameterValue("b3Q")->load();

        auto hpCoeffs = IIRCoeffs::makeHighPass(currentSampleRate, lowCut, 0.707f);
        auto lpCoeffs = IIRCoeffs::makeLowPass(currentSampleRate, highCut, 0.707f);
        auto p1Coeffs = IIRCoeffs::makePeakFilter(currentSampleRate, b1F, b1Q, b1G);
        auto p2Coeffs = IIRCoeffs::makePeakFilter(currentSampleRate, b2F, b2Q, b2G);
        auto p3Coeffs = IIRCoeffs::makePeakFilter(currentSampleRate, b3F, b3Q, b3G);

        constexpr int numBins = SpectralSuppressorEngine::numBins;
        for (int b = 0; b < numBins; ++b)
        {
            double freq = std::clamp((static_cast<double>(b) * currentSampleRate) / SpectralSuppressorEngine::fftSize, 20.0, currentSampleRate * 0.499);
            double mag = hpCoeffs->getMagnitudeForFrequency(freq, currentSampleRate)
                       * lpCoeffs->getMagnitudeForFrequency(freq, currentSampleRate)
                       * p1Coeffs->getMagnitudeForFrequency(freq, currentSampleRate)
                       * p2Coeffs->getMagnitudeForFrequency(freq, currentSampleRate)
                       * p3Coeffs->getMagnitudeForFrequency(freq, currentSampleRate);
            sidechainWeightsLinear[b] = static_cast<float>(mag);
            sidechainWeightsDB[b]     = juce::Decibels::gainToDecibels(static_cast<float>(mag), -36.0f);
        }
    }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        updateSidechainWeights();

        float depthNorm  = apvts.getRawParameterValue("depth")->load() / 10.0f;
        float detailNorm = apvts.getRawParameterValue("detail")->load() / 10.0f;
        float attackMs   = apvts.getRawParameterValue("attack")->load();
        float releaseMs  = apvts.getRawParameterValue("release")->load();
        bool hardMode    = apvts.getRawParameterValue("hardMode")->load() > 0.5f;
        bool deltaMode   = apvts.getRawParameterValue("delta")->load() > 0.5f;
        bool bypassed    = apvts.getRawParameterValue("bypass")->load() > 0.5f;
        float mix        = bypassed ? 0.0f : (apvts.getRawParameterValue("mix")->load() / 100.0f);
        float outGainLin = juce::Decibels::decibelsToGain(apvts.getRawParameterValue("outGain")->load());

        int numChannels = std::min(buffer.getNumChannels(), 2);
        int numSamples  = buffer.getNumSamples();

        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* channelData = buffer.getWritePointer(ch);
            for (int s = 0; s < numSamples; ++s)
            {
                channelData[s] = engines[ch].processSample(channelData[s], sidechainWeightsLinear,
                                                           depthNorm, detailNorm, attackMs, releaseMs,
                                                           hardMode, deltaMode && !bypassed, mix) * outGainLin;
            }
        }
    }

    void getVisualData(std::array<float, SpectralSuppressorEngine::numBins>& outReductionDB,
                       std::array<float, SpectralSuppressorEngine::numBins>& outDepthCurveDB) const
    {
        const auto& redL = engines[0].getVisualReductionDB();
        const auto& redR = engines[1].getVisualReductionDB();
        for (size_t i = 0; i < SpectralSuppressorEngine::numBins; ++i)
        {
            outReductionDB[i]  = std::min(redL[i], redR[i]);
            outDepthCurveDB[i] = sidechainWeightsDB[i];
        }
    }

    double getCurrentSampleRate() const { return currentSampleRate; }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override
    {
        auto state = apvts.copyState();
        std::unique_ptr<juce::XmlElement> xml(state.createXml());
        copyXmlToBinary(*xml, destData);
    }

    void setStateInformation(const void* data, int sizeInBytes) override
    {
        std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
        if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
    }

    juce::AudioProcessorValueTreeState apvts;

private:
    std::array<SpectralSuppressorEngine, 2> engines;
    std::array<float, SpectralSuppressorEngine::numBins> sidechainWeightsLinear {};
    std::array<float, SpectralSuppressorEngine::numBins> sidechainWeightsDB {};
    double currentSampleRate = 44100.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DynamicResonanceSuppressorAudioProcessor)
};

// ==============================================================================
// 3. INTERACTIVE GUI EDITOR
// ==============================================================================
class DynamicResonanceSuppressorAudioProcessorEditor : public juce::AudioProcessorEditor,
                                                       private juce::Timer
{
public:
    explicit DynamicResonanceSuppressorAudioProcessorEditor(DynamicResonanceSuppressorAudioProcessor& p)
        : AudioProcessorEditor(&p), audioProcessor(p)
    {
        setSize(920, 540);

        auto setupRotary = [this](juce::Slider& s, juce::Label& l, const juce::String& text, const juce::String& suffix = "")
        {
            s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
            s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 64, 18);
            s.setTextValueSuffix(suffix);
            s.setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xff2c3e55));
            s.setColour(juce::Slider::thumbColourId, juce::Colours::white);
            s.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xff1e2b3c));
            s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
            addAndMakeVisible(s);

            l.setText(text, juce::dontSendNotification);
            l.setJustificationType(juce::Justification::centred);
            l.setColour(juce::Label::textColourId, juce::Colour(0xff1e2b3c));
            l.setFont(juce::FontOptions(13.0f, juce::Font::bold));
            addAndMakeVisible(l);
        };

        setupRotary(depthSlider,   depthLabel,   "DEPTH");
        setupRotary(detailSlider,  detailLabel,  "DETAIL");
        setupRotary(attackSlider,  attackLabel,  "ATTACK",  " ms");
        setupRotary(releaseSlider, releaseLabel, "RELEASE", " ms");
        setupRotary(mixSlider,     mixLabel,     "MIX",     " %");
        setupRotary(outGainSlider, outGainLabel, "OUT GAIN"," dB");

        auto setupToggle = [this](juce::TextButton& b)
        {
            b.setClickingTogglesState(true);
            b.setColour(juce::TextButton::buttonColourId,   juce::Colour(0xff8fa8c8));
            b.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff203044));
            b.setColour(juce::TextButton::textColourOffId,  juce::Colour(0xff1e2b3c));
            b.setColour(juce::TextButton::textColourOnId,   juce::Colours::white);
            addAndMakeVisible(b);
        };

        setupToggle(softHardButton);
        setupToggle(deltaButton);
        setupToggle(bypassButton);

        depthAttach   = std::make_unique<SliderAttachment>(audioProcessor.apvts, "depth",   depthSlider);
        detailAttach  = std::make_unique<SliderAttachment>(audioProcessor.apvts, "detail",  detailSlider);
        attackAttach  = std::make_unique<SliderAttachment>(audioProcessor.apvts, "attack",  attackSlider);
        releaseAttach = std::make_unique<SliderAttachment>(audioProcessor.apvts, "release", releaseSlider);
        mixAttach     = std::make_unique<SliderAttachment>(audioProcessor.apvts, "mix",     mixSlider);
        outGainAttach = std::make_unique<SliderAttachment>(audioProcessor.apvts, "outGain", outGainSlider);

        hardAttach    = std::make_unique<ButtonAttachment>(audioProcessor.apvts, "hardMode", softHardButton);
        deltaAttach   = std::make_unique<ButtonAttachment>(audioProcessor.apvts, "delta",    deltaButton);
        bypassAttach  = std::make_unique<ButtonAttachment>(audioProcessor.apvts, "bypass",   bypassButton);

        startTimerHz(30);
    }

    ~DynamicResonanceSuppressorAudioProcessorEditor() override { stopTimer(); }

    void timerCallback() override
    {
        audioProcessor.getVisualData(visualReductionDB, visualDepthCurveDB);
        repaint(graphBounds.toNearestInt());
    }

    float freqToX(float freq, juce::Rectangle<float> b) const
    {
        float norm = std::log(std::clamp(freq, 20.0f, 20000.0f) / 20.0f) / std::log(20000.0f / 20.0f);
        return b.getX() + norm * b.getWidth();
    }

    float xToFreq(float x, juce::Rectangle<float> b) const
    {
        float norm = std::clamp((x - b.getX()) / b.getWidth(), 0.0f, 1.0f);
        return 20.0f * std::pow(20000.0f / 20.0f, norm);
    }

    float dbToY(float db, juce::Rectangle<float> b) const
    {
        float norm = juce::jmap(std::clamp(db, -24.0f, 24.0f), 24.0f, -24.0f, 0.0f, 1.0f);
        return b.getY() + norm * b.getHeight();
    }

    float yToDb(float y, juce::Rectangle<float> b) const
    {
        float norm = std::clamp((y - b.getY()) / b.getHeight(), 0.0f, 1.0f);
        return juce::jmap(norm, 0.0f, 1.0f, 24.0f, -24.0f);
    }

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colour(0xffa9c1df));

        auto leftPanel = getLocalBounds().removeFromLeft(200).toFloat();
        g.setColour(juce::Colour(0xff9bb5d6));
        g.fillRect(leftPanel);

        g.setColour(juce::Colour(0xff203044));
        g.setFont(juce::FontOptions(24.0f, juce::Font::bold));
        g.drawText("soothe-dsp", 20, 16, 160, 28, juce::Justification::centredLeft);
        g.setFont(juce::FontOptions(11.0f));
        g.drawText("DYNAMIC RESONANCE SUPPRESSOR", 20, 42, 175, 16, juce::Justification::centredLeft);

        g.setColour(juce::Colour(0xff7d8ea6));
        g.fillRoundedRectangle(graphBounds, 4.0f);

        float zeroDbY = dbToY(0.0f, graphBounds);
        juce::Rectangle<float> lowerHalf(graphBounds.getX(), zeroDbY, graphBounds.getWidth(), graphBounds.getBottom() - zeroDbY);
        g.setColour(juce::Colour(0xff5e7291));
        g.fillRect(lowerHalf);

        g.setColour(juce::Colours::white.withAlpha(0.18f));
        g.setFont(juce::FontOptions(10.0f));
        const float gridFreqs[] = { 50.0f, 100.0f, 200.0f, 500.0f, 1000.0f, 2000.0f, 5000.0f, 10000.0f };
        for (float f : gridFreqs)
        {
            float x = freqToX(f, graphBounds);
            g.drawVerticalLine(static_cast<int>(x), graphBounds.getY(), graphBounds.getBottom());
            juce::String label = (f >= 1000.0f) ? juce::String(f / 1000.0f, 0) + "k" : juce::String(static_cast<int>(f));
            g.drawText(label, static_cast<int>(x) - 15, static_cast<int>(graphBounds.getY()) + 4, 30, 14, juce::Justification::centred);
        }

        g.setColour(juce::Colours::white.withAlpha(0.45f));
        g.drawHorizontalLine(static_cast<int>(zeroDbY), graphBounds.getX(), graphBounds.getRight());

        juce::Path reductionPath;
        reductionPath.startNewSubPath(graphBounds.getX(), zeroDbY);
        double sr = audioProcessor.getCurrentSampleRate();
        constexpr int numBins = SpectralSuppressorEngine::numBins;

        for (int b = 1; b < numBins; ++b)
        {
            float freq = static_cast<float>((b * sr) / SpectralSuppressorEngine::fftSize);
            if (freq < 20.0f || freq > 20000.0f) continue;
            reductionPath.lineTo(freqToX(freq, graphBounds), dbToY(visualReductionDB[b], graphBounds));
        }
        reductionPath.lineTo(graphBounds.getRight(), zeroDbY);
        reductionPath.closeSubPath();

        g.setColour(juce::Colour(0xff1e2d42).withAlpha(0.80f));
        g.fillPath(reductionPath);
        g.setColour(juce::Colour(0xff89b4f8));
        g.strokePath(reductionPath, juce::PathStrokeType(1.5f));

        juce::Path depthPath;
        bool started = false;
        for (int b = 1; b < numBins; ++b)
        {
            float freq = static_cast<float>((b * sr) / SpectralSuppressorEngine::fftSize);
            if (freq < 20.0f || freq > 20000.0f) continue;
            float x = freqToX(freq, graphBounds);
            float y = dbToY(visualDepthCurveDB[b], graphBounds);
            if (!started) { depthPath.startNewSubPath(x, y); started = true; }
            else depthPath.lineTo(x, y);
        }

        g.setColour(juce::Colours::white.withAlpha(0.9f));
        g.strokePath(depthPath, juce::PathStrokeType(2.0f));

        auto drawNode = [&](float freq, float db, const juce::String& name, bool isCut)
        {
            float x = freqToX(freq, graphBounds);
            float y = dbToY(db, graphBounds);
            g.setColour(isCut ? juce::Colour(0xffd0e2ff) : juce::Colours::white);
            g.fillEllipse(x - 6.0f, y - 6.0f, 12.0f, 12.0f);
            g.setColour(juce::Colour(0xff203044));
            g.drawEllipse(x - 6.0f, y - 6.0f, 12.0f, 12.0f, 1.5f);
            g.setFont(juce::FontOptions(10.0f, juce::Font::bold));
            g.setColour(juce::Colours::white);
            g.drawText(name, static_cast<int>(x) - 18, static_cast<int>(y) - 20, 36, 12, juce::Justification::centred);
        };

        drawNode(audioProcessor.apvts.getRawParameterValue("lowCutFreq")->load(),  0.0f, "LC", true);
        drawNode(audioProcessor.apvts.getRawParameterValue("b1Freq")->load(), audioProcessor.apvts.getRawParameterValue("b1Gain")->load(), "B1", false);
        drawNode(audioProcessor.apvts.getRawParameterValue("b2Freq")->load(), audioProcessor.apvts.getRawParameterValue("b2Gain")->load(), "B2", false);
        drawNode(audioProcessor.apvts.getRawParameterValue("b3Freq")->load(), audioProcessor.apvts.getRawParameterValue("b3Gain")->load(), "B3", false);
        drawNode(audioProcessor.apvts.getRawParameterValue("highCutFreq")->load(), 0.0f, "HC", true);
    }

    void resized() override
    {
        softHardButton.setBounds(45, 72, 110, 26);
        depthLabel.setBounds(30, 110, 140, 18);
        depthSlider.setBounds(30, 128, 140, 130);
        detailLabel.setBounds(45, 268, 110, 18);
        detailSlider.setBounds(45, 286, 110, 95);
        attackLabel.setBounds(15, 392, 80, 18);
        attackSlider.setBounds(15, 410, 80, 80);
        releaseLabel.setBounds(105, 392, 80, 18);
        releaseSlider.setBounds(105, 410, 80, 80);

        graphBounds = juce::Rectangle<float>(215.0f, 20.0f, static_cast<float>(getWidth() - 235), static_cast<float>(getHeight() - 95));

        int footerY = getHeight() - 62;
        deltaButton.setBounds(215, footerY + 12, 80, 28);
        mixLabel.setBounds(480, footerY - 4, 70, 16);
        mixSlider.setBounds(470, footerY + 10, 90, 48);
        outGainLabel.setBounds(590, footerY - 4, 70, 16);
        outGainSlider.setBounds(580, footerY + 10, 90, 48);
        bypassButton.setBounds(getWidth() - 105, footerY + 12, 85, 28);
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        if (!graphBounds.contains(e.position)) return;
        struct NodeInfo { float f; float g; };
        std::array<NodeInfo, 5> nodes = {{
            { audioProcessor.apvts.getRawParameterValue("lowCutFreq")->load(),  0.0f },
            { audioProcessor.apvts.getRawParameterValue("b1Freq")->load(),      audioProcessor.apvts.getRawParameterValue("b1Gain")->load() },
            { audioProcessor.apvts.getRawParameterValue("b2Freq")->load(),      audioProcessor.apvts.getRawParameterValue("b2Gain")->load() },
            { audioProcessor.apvts.getRawParameterValue("b3Freq")->load(),      audioProcessor.apvts.getRawParameterValue("b3Gain")->load() },
            { audioProcessor.apvts.getRawParameterValue("highCutFreq")->load(), 0.0f }
        }};

        draggedNodeIndex = -1;
        float minDist = 18.0f;
        for (int i = 0; i < 5; ++i)
        {
            juce::Point<float> pt(freqToX(nodes[i].f, graphBounds), dbToY(nodes[i].g, graphBounds));
            float d = e.position.getDistanceFrom(pt);
            if (d < minDist) { minDist = d; draggedNodeIndex = i; }
        }
    }

    void mouseDrag(const juce::MouseEvent& e) override
    {
        if (draggedNodeIndex < 0) return;
        float newFreq = xToFreq(e.position.x, graphBounds);
        float newGain = yToDb(e.position.y, graphBounds);

        auto setParam = [this](const juce::String& id, float plainValue)
        {
            if (auto* p = audioProcessor.apvts.getParameter(id))
                p->setValueNotifyingHost(p->convertTo0to1(plainValue));
        };

        if (draggedNodeIndex == 0)
            setParam("lowCutFreq", std::clamp(newFreq, 20.0f, 2000.0f));
        else if (draggedNodeIndex == 4)
            setParam("highCutFreq", std::clamp(newFreq, 2000.0f, 20000.0f));
        else
        {
            juce::String prefix = "b" + juce::String(draggedNodeIndex);
            setParam(prefix + "Freq", std::clamp(newFreq, 40.0f, 18000.0f));
            setParam(prefix + "Gain", std::clamp(newGain, -18.0f, 18.0f));
        }
    }

    void mouseUp(const juce::MouseEvent&) override { draggedNodeIndex = -1; }

    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override
    {
        if (!graphBounds.contains(e.position)) return;
        for (int i = 1; i <= 3; ++i)
        {
            juce::String prefix = "b" + juce::String(i);
            float f = audioProcessor.apvts.getRawParameterValue(prefix + "Freq")->load();
            float g = audioProcessor.apvts.getRawParameterValue(prefix + "Gain")->load();
            juce::Point<float> pt(freqToX(f, graphBounds), dbToY(g, graphBounds));

            if (e.position.getDistanceFrom(pt) < 28.0f)
            {
                if (auto* qParam = audioProcessor.apvts.getParameter(prefix + "Q"))
                {
                    float curQ = audioProcessor.apvts.getRawParameterValue(prefix + "Q")->load();
                    float nextQ = std::clamp(curQ + wheel.deltaY * 1.5f, 0.3f, 8.0f);
                    qParam->setValueNotifyingHost(qParam->convertTo0to1(nextQ));
                }
                break;
            }
        }
    }

private:
    DynamicResonanceSuppressorAudioProcessor& audioProcessor;
    juce::Slider depthSlider, detailSlider, attackSlider, releaseSlider, mixSlider, outGainSlider;
    juce::Label depthLabel, detailLabel, attackLabel, releaseLabel, mixLabel, outGainLabel;
    juce::TextButton softHardButton { "SOFT / HARD" }, deltaButton { "DELTA" }, bypassButton { "BYPASS" };

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    std::unique_ptr<SliderAttachment> depthAttach, detailAttach, attackAttach, releaseAttach, mixAttach, outGainAttach;
    std::unique_ptr<ButtonAttachment> hardAttach, deltaAttach, bypassAttach;

    juce::Rectangle<float> graphBounds;
    int draggedNodeIndex = -1;
    std::array<float, SpectralSuppressorEngine::numBins> visualReductionDB {}, visualDepthCurveDB {};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DynamicResonanceSuppressorAudioProcessorEditor)
};

juce::AudioProcessorEditor* DynamicResonanceSuppressorAudioProcessor::createEditor()
{
    return new DynamicResonanceSuppressorAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DynamicResonanceSuppressorAudioProcessor();
}
