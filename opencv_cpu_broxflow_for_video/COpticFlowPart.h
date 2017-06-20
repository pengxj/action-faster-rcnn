#ifndef COpticFlowH
#define COpticFlowH

#include "CVector.h"
#include "CMatrix.h"
#include "CTensor.h"
#include "CTensor4D.h"

class COpticFlow {
public:
  COpticFlow() {}

  static void warpingGeometric(CTensor<float> aFirst, CTensor<float> aSecond,
                               CTensor<float>& aFlow,
                               CTensor<float>& aGeometricU, CTensor<float>& aGeometricV, CTensor<float>& aGeometricConfidence,
                               CMatrix<float>* aEdgeMap = 0,
                               float aSigma = 0.9f, float aAlpha = 80.0f, float aBeta = 2000.0f, float aGamma = 5.0f, float aEdgeWeight = 0.00025f,
                               float aEta = 0.95f, int aFixedpointIterations = 5, int aSORIterations = 5, float aOmega = 1.85f);

  // Visualization of flow fields as color plots
  static void cartesianToRGB (float x, float y, float& R, float& G, float& B);
  static void flowToImage(CTensor<float>& aFlow, CTensor<float>& aImage, float aDivisor = 1.0f);

  // Others
  static void warp(const CTensor<float>& aImage, CTensor<float>& aWarped, CMatrix<bool>& aOutside, CTensor<float>& aFlow);
  static void writeMiddlebury(CTensor<float>& aFlow, const char* aFilename);
  static void readMiddlebury(const char* aFilename, CTensor<float>& aFlow);
  static float angularError(CTensor<float>& aFlow, CTensor<float>& aCorrect);

private:
  static void diffusivity(CTensor<float>& aFlow, CTensor<float>& aFlowIncrement, CMatrix<float>& aResultX, CMatrix<float>& aResultY, CMatrix<float>* aEdgeMap = 0, float aEdgeWeight = 0.00025f);
  static void nonlinearIteration(CTensor<float>& aFirst, CTensor<float>& aSecond, CMatrix<float>& aConfidence, CTensor<float>& aGeometricU, CTensor<float>& aGeometricV, CTensor<float>& aGeometricConfidence, CTensor<float>& aFlow, CMatrix<float>* aEdgeMap, float aAlpha, float aBeta, float aGamma, float aEdgeWeight, int aFixedPointIterations, int aSORIterations, float aOmega);
};

#endif

