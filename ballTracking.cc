/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp" // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "emdw.hpp"
#include "discretetable.hpp"
#include "combinations.hpp"

// standard headers
#include <iostream> // cout, endl, flush, cin, cerr
#include <cctype>   // toupper
#include <string>   // string
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

#include "prlite_genmat.hpp"
#include "sqrtmvg.hpp"

// In order to use the built in Inference packages
// standard library headers
#include <vector> // vector

// emdw headers
#include "clustergraph.hpp"
#include "lbp_cg.hpp"
#include "lbu_cg.hpp"
#include "messagequeue.hpp"

using namespace std;
using namespace emdw;

// ##################################################################
//  Some example code. To compile this, go to the emdw/build
//  directory and do a:
//  cmake ../; make -j7 example
//  To run this while in the build directory, do a:
//  src/pmr/example
//
//  For your own stuff, make a copy of this one to start with. Then
//  edit the CMakeLists.txt (also in this directory) by adding your
//  new target in the same way as this example.
// ##################################################################

int main(int, char *argv[])
{

  // NOTE: this activates logging and unit tests
  initLogging(argv[0]);
  prlite::TestCase::runAllTests();

  try
  {

    //*********************************************************
    // Some random generator seeding. Just keep this as is
    //*********************************************************

    unsigned seedVal = emdw::randomEngine.getSeedVal();
    cout << seedVal << endl;
    emdw::randomEngine.setSeedVal(seedVal);

    //*********************************************************
    // Predefine some types and constants
    //*********************************************************
    typedef SqrtMVG SG; // Shorthand for SqrtMVG

    int frames = 2; // Number of framess which are being tracked, fps*video length

    // Examples
    vector<double> meansx1 = {230.0, 235.0}; // Mean x positions for IPA1
    vector<double> meansy1 = {250.0, 260.0}; // Mean y positions for IPA1

    vector<double> meansx2 = {225.0, 232.0}; // Mean x positions for IPA2
    vector<double> meansy2 = {256.0, 263.0}; // Mean y positions for IPA2

    prlite::RowMatrix<double> cov_Ipa(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa(0, 0) = cov_Ipa(1, 1) = 25.0;
    cov_Ipa(0, 1) = cov_Ipa(1, 0) = 0.0;

    prlite::ColVector<double> mean_trans(4); // mean vector for the transition factor

    // Transition factor: (x_prev, y_prev) -> (x, y)
    prlite::RowMatrix<double> cov_trans(4, 4);
    for (int d = 0; d < 4; d++)
    {
      for (int e = 0; e < 4; e++)
        cov_trans(d, e) = 0.0; // zero everything
      cov_trans(d, d) = 50.0;  // set diagonal
    } // This can be adjusted as needed

    // Start the for loop to iterate through the frames, However might do 2 forms of inference BU and BP

    map<RVIdType, AnyType> obvs;

    unsigned int x_prev = 0;
    unsigned int y_prev = 1;

    for (int i = 0; i < frames; i++)
    {

      unsigned int x = 2 * i;
      unsigned int y = 2 * i + 1;

      //Image Processing Algorithm 1 (IPA1) code

      prlite::ColVector<double> mn1(2); // mean vector 1
      mn1[0] = meansx1[i];
      mn1[1] = meansy1[i];
      rcptr<Factor> alg_1(uniqptr<SG>(new SG({x, y}, mn1, cov_Ipa))); // Setting up the Guassian factor for IPA1

      //Image Processing Algorithm 2 (IPA2) code

      prlite::ColVector<double> mn2(2); // Mean vector 2
      mn2[0] = meansx2[i];
      mn2[1] = meansy2[i];
      rcptr<Factor> alg_2(uniqptr<SG>(new SG({x, y}, mn2, cov_Ipa))); // Setting up the Guassian factor for IPA2

      vector<rcptr<Factor>> factors;

      if (i == 0)
      {
        factors = {alg_1, alg_2}; // Create a vector of factors for the first frame
      }
      else
      {

        rcptr<Factor> trans(new SqrtMVG({x_prev, y_prev, x, y}, mean_trans, cov_trans)); // Transition factor connecting the previous frame's (x_prev, y_prev) to the current frame's (x, y)

        factors = {alg_1, alg_2, trans}; // Create a vector of factors for the subsequent frames
      }

      ClusterGraph cg(ClusterGraph::LTRIP, factors, obvs); // Create a cluster graph with the factors

      map<Idx2z, rcptr<Factor>> msgs; // Calibrate via loopy belief propagation (Shafer–Shenoy)
      MessageQueue msgQ;
      unsigned nMsgs = loopyBP_CG(cg, msgs, msgQ);
      cout << "Sent " << nMsgs << " messages before convergence\n";

      rcptr<Factor> qPtr = queryLBP_CG(cg, msgs, {x, y})->normalize(); // Query the (x,y) belief for this frame

      auto qPtr_Sqrt = static_cast<SqrtMVG *>(qPtr.get());

      cout << "LBP belief mean for frame " << (i + 1) << ": " << qPtr_Sqrt->getMean() << endl;

      auto mean = qPtr_Sqrt->getMean();
      mean_trans[0] = mean[0]; // expected delta x prev
      mean_trans[1] = mean[1]; // expected delta y prev
      mean_trans[2] = mean[0]; // expected delta x (for the second observation)
      mean_trans[3] = mean[1]; // expected delta y (for the second observation)
      // These are for the transistion factor, which is used to get the next state based on the previous state

      x_prev = x; // Update previous x and y for the next iteration
      y_prev = y;
    }

    return 0; // tell the world that all is fine
  } // try

  catch (char msg[])
  {
    cerr << msg << endl;
  } // catch

  // catch (char const* msg) {
  //   cerr << msg << endl;
  // } // catch

  catch (const string &msg)
  {
    cerr << msg << endl;
    throw;
  } // catch

  catch (const exception &e)
  {
    cerr << "Unhandled exception: " << e.what() << endl;
    throw e;
  } // catch

  catch (...)
  {
    cerr << "An unknown exception / error occurred\n";
    throw;
  } // catch

} // main
