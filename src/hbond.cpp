/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2023- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements gmx::analysismodules::Hbond2.
 *
 * \author Sergey Gorelov <gorelov_sv@pnpi.nrcki.ru>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include <algorithm>
#include <iostream>
#include <set>
#include <utility>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/histogram.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/math/units.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/optionfiletype.h"
#include "gromacs/options/options.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/cmdlinerunner.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"


using namespace gmx;

struct HBond
{
    HBond(int d, int a, int h = -1) : donor(d), acceptor(a), hydrogen(h) {}
    int  donor;
    int  acceptor;
    int  hydrogen;
    bool operator<(const HBond& other) const;
};

bool HBond::operator<(const HBond& other) const
{
    return (this->donor < other.donor) || (this->donor == other.donor && this->acceptor < other.acceptor)
           || (this->donor == other.donor && this->acceptor == other.acceptor
               && this->hydrogen < other.hydrogen);
}

//! Structure that contains storage information from different frames.
struct HbondStorageFrame
{
    HbondStorageFrame(int frnr, std::vector<HBond> data) :
        frameNumber_(frnr), hbondData_(std::move(data))
    {
    }
    //! Frame number.
    int frameNumber_ = 0;
    //! Frame hbond data.
    std::vector<HBond> hbondData_;
};

/*! \brief
 * Class that stores frame information in storage and, upon request, can return it.
 */
class HbondStorage
{
public:
    /*! \brief
     * Function that stores frame information in storage.
     */
    void addData(int frnr, const std::vector<HBond>& data);
    /*! \brief
     * Function that returns frame information from storage.
     */
    const std::vector<HbondStorageFrame>& getData() const;

private:
    /*! \brief
     * Vector that contains information from different frames.
     */
    std::vector<HbondStorageFrame> data_;
};

void HbondStorage::addData(int frnr, const std::vector<HBond>& data)
{
    data_.emplace_back(frnr, data);
}

const std::vector<HbondStorageFrame>& HbondStorage::getData() const
{
    return data_;
}

constexpr std::size_t sc_maxHydogens = 4;

struct t_acceptor
{
    t_acceptor(int atomIndex) : ai(atomIndex) {}
    // Acceptor Atom Qndex
    int  ai;
    bool isAlsoDonor = false;
};

struct t_donor
{
    t_donor(int atomIndex) : ai(atomIndex) {}
    // Donor Atom Index
    int ai;
    // Associated Hydrogen Atoms Indices
    //    gmx::FixedCapacityVector<int, sc_maxHydogens> h_atoms;
    std::vector<int> h_atoms;
    void             add_hydrogen(int hi);
};

void t_donor::add_hydrogen(const int hi)
{
    if (h_atoms.size() >= sc_maxHydogens)
    {
        GMX_THROW(InconsistentInputError("Oops! Too much hydrogens to handle."));
    }
    h_atoms.push_back(hi);
}

struct t_info
{
    std::vector<t_acceptor> acceptors;
    std::vector<t_donor>    donors;
    const Selection*        selPtr = nullptr;
    std::vector<int>        atomIndices;
};

class Hbond : public TrajectoryAnalysisModule
{
public:
    Hbond();
    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    static bool isInSelection(int ai, const std::vector<int>& selVector);
    /*! \brief Check partial overlapping between two parsed selections
     * and throw an error when groups are overlapping but are not equal. */
    void checkOverlap() const;
    void searchAcceptors(const TopologyInformation& top, t_info* selTool, const std::vector<int>* sel);
    void searchDonors(const TopologyInformation& top, t_info* selTool, const std::vector<int>* sel);
    void linkDA(t_info* selTool);
    void prepareForAnalysis(const TrajectoryAnalysisSettings& settings);
    std::set<HBond> prepareFrameData(const std::vector<HbondStorageFrame>& data) const;
    std::set<HBond> prepareFrameData(const std::vector<HBond>& data) const;
    Selection       refSel_, targetSel_;
    std::string     fnmHbondOut_ = "hbond";
    std::string     fnmHbnumOut_;
    std::string     fnmHbdistOut_;
    std::string     fnmHbangOut_;
    std::string     fnmHbdanOut_;
    t_info          refInfo, targetInfo;
    bool            isTwoDiffGroups_  = false;
    bool            acceptN           = true;
    bool            perFrame_         = false;
    bool            mergeHydrogens_   = false;
    real            cutoff_           = 0.35;
    const float     c_rMaxNM_         = 0.35;
    const float     c_angleMaxDegree_ = 30;
    HbondStorage    storage_;

    AnalysisData                             hbnum_;
    AnalysisData                             distances_;
    AnalysisData                             angles_;
    AnalysisData                             dan_;
    AnalysisDataModulePointer                plotter_;
    AnalysisDataSimpleHistogramModulePointer histogramModuleDist_;
    AnalysisDataSimpleHistogramModulePointer histogramModuleAng_;
};

Hbond::Hbond() :
    histogramModuleDist_(new AnalysisDataSimpleHistogramModule),
    histogramModuleAng_(new AnalysisDataSimpleHistogramModule)
{
    registerAnalysisDataset(&hbnum_, "hbnum");

    registerAnalysisDataset(&distances_, "hbdist");
    distances_.addModule(histogramModuleDist_);
    registerBasicDataset(&histogramModuleDist_->averager(), "histogram_dist");

    registerAnalysisDataset(&angles_, "hbang");
    angles_.addModule(histogramModuleAng_);
    registerBasicDataset(&histogramModuleAng_->averager(), "histogram_ang");

    registerAnalysisDataset(&dan_, "hbdan");
}

void Hbond::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] allows using geometric definition of hydrogen bonds to define them "
        "throughout the structure.[PAR]"
        "[TT]-r[tt] specifies reference selection, relative to which the search for hydrogen bonds "
        "in target selection will develop. Note that all atoms in reference and target selections "
        "should be either absolutely identical or non-overlapping at all.[PAR]"
        "[TT]-t[tt] specifies target selection, relative to which the search for hydrogen bonds in "
        "reference selection will develop. Note that all atoms in reference and target selections "
        "should be either absolutely identical or non-overlapping at all.[PAR]"
        "[TT]-m[tt] forces to merge together information in output index file about hydrogen bonds "
        "if they differ only in hydrogen indicies. This also means that information about hydrogen "
        "atoms in the hydrogen bonds would not be written in output index file at all.[PAR]"
        "[TT]-pf[tt] forces to write hydrogen bonds for each frame separatly instead of writing "
        "hydrogen bonds for the whole system. Each information about hydrogen bonds in new frame "
        "will be stored in its own section of the output index file.[PAR]"
        "[TT]-an[tt] forces to accept nitrogen atoms as acceptors of hydrogen bond. If this option "
        "is set to 'false' state, only oxygen atoms will be considered as acceptors.[PAR]"
        "[TT]-cutoff[tt] is a real value that defines distance from donor to acceptor "
        "(and vise versa) that used in neighbor search. Minimum (and also recommended) value is "
        "0.35.[PAR]"
        "[TT]-num[tt] allows you to get a plot of the number of hydrogen bonds as a function of "
        "time at the output.[PAR]"
        "[TT]-dist[tt] allows you to get a plot of the distance distribution of all hydrogen bonds "
        "at the output.[PAR]"
        "[TT]-ang[tt] allows you to get a plot of the angular distribution of all hydrogen bonds "
        "at the output.[PAR]"
        "[TT]-dan[tt] allows you to get a plot of the number of analyzed donors and acceptors for "
        "each frame at the output.[PAR]"
    };

    options->addOption(
            FileNameOption("o")
                    .filetype(OptionFileType::Index)
                    .outputFile()
                    .required()
                    .store(&fnmHbondOut_)
                    .defaultBasename("hbond")
                    .description("Index file that contains selected groups', acceptors', "
                                 "donors' and hydrogens' indicies and hydroden bonds "
                                 "pairs between or within selected groups."));
    options->addOption(FileNameOption("num")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&fnmHbnumOut_)
                               .defaultBasename("hbnum")
                               .description("Number of hydrogen bonds as a function of time."));
    options->addOption(FileNameOption("dist")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&fnmHbdistOut_)
                               .defaultBasename("hbdist")
                               .description("Distance distribution of all hydrogen bonds."));
    options->addOption(FileNameOption("ang")
                               .filetype(OptionFileType::Plot)
                               .outputFile()
                               .store(&fnmHbangOut_)
                               .defaultBasename("hbang")
                               .description("Angle distribution of all hydrogen bonds."));
    options->addOption(
            FileNameOption("dan")
                    .filetype(OptionFileType::Plot)
                    .outputFile()
                    .store(&fnmHbdanOut_)
                    .defaultBasename("hbdan")
                    .description("Number of donors and acceptors analyzed for each frame."));
    options->addOption(SelectionOption("r").store(&refSel_).required().description(
            "Reference selection, relative to which the search for hydrogen bonds in target "
            "selection will develop."));
    options->addOption(
            SelectionOption("t").store(&targetSel_).required().description("Target selection, relative to which the search for hydrogen bonds in reference selection will develop."));
    options->addOption(
            BooleanOption("m").store(&mergeHydrogens_).defaultValue(false).description("Merge together information about hydrogen bonds if they differ only in hydrogen indicies."));
    options->addOption(
            BooleanOption("pf").store(&perFrame_).defaultValue(false).description("Write hydrogen bonds for each frame separatly instead of writing hydrogen bonds for the whole system."));
    options->addOption(BooleanOption("an").store(&acceptN).defaultValue(true).description(
            "Accept nitrogen atoms as acceptors of hydrogen bond."));
    options
            ->addOption(RealOption("cutoff").store(&cutoff_).required().defaultValue(0.35).description("Distance from donor to acceptor (and vise versa) that used in neighbor search. Must be >= 0.35."));
    settings->setHelpText(desc);

    settings->setFlag(TrajectoryAnalysisSettings::efRequireTop);
}

void Hbond::optionsFinished(TrajectoryAnalysisSettings* /* settings */)
{
    if (cutoff_ < static_cast<real>(0.35))
    {
        GMX_THROW(InconsistentInputError("Invalid cutoff value. It must be >= 0.35."));
    }
}

bool Hbond::isInSelection(int ai, const std::vector<int>& selVector)
{
    return std::find(selVector.begin(), selVector.end(), ai) != selVector.end();
}

void Hbond::checkOverlap() const
{
    printf("Checking for overlap in atoms between %s (%d atoms) and %s (%d atoms)\n",
           refSel_.name(),
           refSel_.atomCount(),
           targetSel_.name(),
           targetSel_.atomCount());
    bool isOverlapping = false;

    if (refInfo.atomIndices[0] == targetInfo.atomIndices[0])
    {
        if (refInfo.atomIndices != targetInfo.atomIndices)
        {
            isOverlapping = true;
        }
    }
    else
    {
        auto i = refInfo.atomIndices.begin(), j = targetInfo.atomIndices.begin();
        while ((i != refInfo.atomIndices.end() && j != targetInfo.atomIndices.end()) && !isOverlapping)
        {
            if (*i == *j)
            {
                isOverlapping = true;
            }
            else if (*i < *j)
            {
                ++i;
            }
            else
            {
                ++j;
            }
        }
    }

    GMX_RELEASE_ASSERT(not isOverlapping,
                       (std::string("Partial overlap between groups '") + refSel_.name() + "' and '"
                        + targetSel_.name() + "'")
                               .c_str());
}

void Hbond::searchAcceptors(const TopologyInformation& top, t_info* selTool, const std::vector<int>* sel)
{
    for (auto ai = sel->begin(); ai != sel->end(); ++ai)
    {
        std::string element         = top.atoms()->atom[static_cast<std::size_t>(*ai)].elem;
        bool        acceptorElement = (element == "O" || (acceptN && element == "N"));
        if (acceptorElement && isInSelection(*ai, *sel))
        {
            selTool->acceptors.emplace_back(*ai);
        }
    }

    if (!selTool->acceptors.empty())
    {
        std::sort(selTool->acceptors.begin(),
                  selTool->acceptors.end(),
                  [](t_acceptor a1, t_acceptor a2) { return a1.ai < a2.ai; });
    }
    else
    {
        fprintf(stderr, "WARNING: Selection '%s' has no acceptors!\n", selTool->selPtr->name());
    }
}

void Hbond::searchDonors(const TopologyInformation& top, t_info* selTool, const std::vector<int>* sel)
{
    std::vector<std::pair<int, int>> dhUnsorted;
    for (t_functype func_type = 0; (func_type < F_NRE); ++func_type)
    {
        const InteractionList* interaction = &(top.expandedTopology()->idef.il[func_type]);
        if (func_type == F_POSRES || func_type == F_FBPOSRES)
        {
            /* We don't need posre interactions for hbonds anyway.*/
            continue;
        }
        for (std::size_t i = 0; i < interaction->iatoms.size();
             i +=
             interaction_function[top.expandedTopology()->idef.functype[interaction->iatoms[i]]].nratoms + 1)
        {
            /* next function */
            if (func_type != top.expandedTopology()->idef.functype[interaction->iatoms[i]])
            {
                fprintf(stderr, "Error in func_type '%s'", interaction_function[func_type].longname);
                continue;
            }
            if (func_type == F_SETTLE)
            {
                int nr1 = interaction->iatoms[i + 1]; // DONOR
                int nr2 = interaction->iatoms[i + 2]; // HYDRO 1
                int nr3 = interaction->iatoms[i + 3]; // HYDRO 2
                if (isInSelection(nr1, *sel))
                {
                    if (isInSelection(nr2, *sel))
                    {
                        dhUnsorted.emplace_back(nr1, nr2);
                    }
                    if (isInSelection(nr3, *sel))
                    {
                        dhUnsorted.emplace_back(nr1, nr3);
                    }
                }
            }
            else if (interaction_function[(func_type)].nratoms == 2
                     && (interaction_function[(func_type)].flags & IF_CHEMBOND))
            {
                for (std::size_t j = 0; j < 2; ++j)
                {
                    int         nr1      = interaction->iatoms[i + 1 + j]; // POTENTIAL HYDRO
                    int         nr2      = interaction->iatoms[i + 2 - j]; // POTENTIAL DONOR
                    std::string element1 = top.atoms()->atom[static_cast<std::size_t>(nr1)].elem;
                    std::string element2 = top.atoms()->atom[static_cast<std::size_t>(nr2)].elem;
                    if (element1 == "H" && (element2 == "O" || element2 == "N"))
                    {
                        if (isInSelection(nr1, *sel) && isInSelection(nr2, *sel))
                        {
                            dhUnsorted.emplace_back(nr2, nr1);
                        }
                    }
                }
            }
        }
    }
    if (!dhUnsorted.empty())
    {
        std::sort(dhUnsorted.begin(), dhUnsorted.end(), [](std::pair<int, int> d1, std::pair<int, int> d2) {
            return d1.first < d2.first || (d1.first == d2.first && d1.second < d2.second);
        });

        int memoryIndex = dhUnsorted.begin()->first - 1;

        for (const auto& pi : dhUnsorted)
        {
            if (memoryIndex != pi.first)
            {
                selTool->donors.emplace_back(pi.first);
                memoryIndex = pi.first;
            }
            selTool->donors.back().add_hydrogen(pi.second);
        }
    }
    else
    {
        fprintf(stderr, "WARNING: Selection '%s' has no donors!\n", selTool->selPtr->name());
    }
}

void Hbond::linkDA(t_info* selTool)
{
    if (!selTool->acceptors.empty() && !selTool->donors.empty())
    {

        auto i = selTool->acceptors.begin();
        auto j = selTool->donors.begin();

        while (i != selTool->acceptors.end() && j != selTool->donors.end())
        {
            if (i->ai == j->ai)
            {
                i->isAlsoDonor = true;
                if (i != selTool->acceptors.end())
                {
                    ++i;
                }
                else if (j != selTool->donors.end())
                {
                    ++j;
                }
                else
                {
                    break;
                }
            }
            else if (i->ai < j->ai)
            {
                ++i;
            }
            else
            {
                ++j;
            }
        }
    }
    else
    {
        GMX_THROW(InconsistentInputError(
                "Selection '" + std::string(selTool->selPtr->name())
                + "' has no donors AND has no acceptors! Nothing to be done."));
    }
}


void Hbond::prepareForAnalysis(const TrajectoryAnalysisSettings& settings)
{
    if (!fnmHbnumOut_.empty())
    {
        hbnum_.setColumnCount(0, 1);
        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(settings.plotSettings()));
        plotm->setFileName(fnmHbnumOut_);
        plotm->setTitle("Number of hydrogen bonds");
        plotm->setXAxisIsTime();
        plotm->setYLabel("Hbonds");
        plotm->appendLegend("Hydrogen bonds");
        hbnum_.addModule(plotm);
    }
    if (!fnmHbdistOut_.empty())
    {
        const double histogramMin = 0;
        const double histogramMax = 0.35;
        histogramModuleDist_->init(
                histogramFromRange(histogramMin, histogramMax).binWidth(0.005).includeAll());

        distances_.setColumnCount(0, ((refInfo.donors.size() + targetInfo.donors.size()) * sc_maxHydogens));

        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(settings.plotSettings()));
        plotm->setFileName(fnmHbdistOut_);
        plotm->setTitle("Hydrogen Bond Distance Distribution Histogram");
        plotm->setXLabel("Distance (nm)");
        plotm->setYLabel("Probability");
        plotm->setXFormat(10, 4);
        plotm->setYFormat(10, 4);
        histogramModuleDist_->averager().addModule(plotm);
    }
    if (!fnmHbangOut_.empty())
    {
        const double histogramMin = 0;
        const double histogramMax = 30;
        histogramModuleAng_->init(histogramFromRange(histogramMin, histogramMax).binWidth(1.0).includeAll());

        angles_.setColumnCount(0, ((refInfo.donors.size() + targetInfo.donors.size()) * sc_maxHydogens));

        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(settings.plotSettings()));
        plotm->setFileName(fnmHbangOut_);
        plotm->setTitle("Hydrogen Bond Angle Distribution Histogram");
        plotm->setXLabel("Hydrogen - Donor - Acceptor Angle (degrees)");
        plotm->setYLabel("Probability");
        plotm->setXFormat(10, 1);
        plotm->setYFormat(10, 7);
        histogramModuleAng_->averager().addModule(plotm);
    }
    if (!fnmHbdanOut_.empty())
    {
        dan_.setColumnCount(0, 2);
        AnalysisDataPlotModulePointer plotm(new AnalysisDataPlotModule(settings.plotSettings()));
        plotm->setFileName(fnmHbdanOut_);
        plotm->setTitle("Number of Donors and Acceptors");
        plotm->setXAxisIsTime();
        plotm->setYLabel("Donors/Acceptors");
        plotm->setXFormat(10, 0);
        plotm->setYFormat(10, 0);
        plotm->appendLegend("Number of Donors");
        plotm->appendLegend("Number of Acceptors");
        dan_.addModule(plotm);
    }
}

void Hbond::initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top)
{
    if (!refSel_.isValid() || !targetSel_.isValid() || refSel_.atomCount() == 0
        || targetSel_.atomCount() == 0)
    {
        GMX_THROW(InvalidInputError("Invalid selection(s). Nothing to be done."));
    }

    refInfo.atomIndices.resize(refSel_.atomIndices().size());
    targetInfo.atomIndices.resize(targetSel_.atomIndices().size());
    std::copy(refSel_.atomIndices().begin(), refSel_.atomIndices().end(), refInfo.atomIndices.begin());
    std::copy(targetSel_.atomIndices().begin(),
              targetSel_.atomIndices().end(),
              targetInfo.atomIndices.begin());

    if (!refSel_.hasSortedAtomIndices())
    {
        std::sort(refInfo.atomIndices.begin(), refInfo.atomIndices.end());
    }

    if (!targetSel_.hasSortedAtomIndices())
    {
        std::sort(targetInfo.atomIndices.begin(), targetInfo.atomIndices.end());
    }

    checkOverlap();

    if (refInfo.atomIndices[0] != targetInfo.atomIndices[0])
    {
        isTwoDiffGroups_ = true;
    }

    refInfo.selPtr    = &refSel_;
    targetInfo.selPtr = &targetSel_;
    searchAcceptors(top, &refInfo, &(refInfo.atomIndices));
    searchDonors(top, &refInfo, &(refInfo.atomIndices));
    linkDA(&refInfo);
    if (isTwoDiffGroups_)
    {
        searchAcceptors(top, &targetInfo, &(targetInfo.atomIndices));
        searchDonors(top, &targetInfo, &(targetInfo.atomIndices));
        linkDA(&targetInfo);
    }

    prepareForAnalysis(settings);
}


void Hbond::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata)
{
    AnalysisDataHandle    dhHbnum       = pdata->dataHandle(hbnum_);
    AnalysisDataHandle    dhDist        = pdata->dataHandle(distances_);
    AnalysisDataHandle    dhAng         = pdata->dataHandle(angles_);
    AnalysisDataHandle    dhDan         = pdata->dataHandle(dan_);
    std::size_t           distIterator  = 0;
    std::size_t           angleIterator = 0;
    std::set<std::size_t> donors;
    std::set<std::size_t> acceptors;


    if (!fnmHbnumOut_.empty())
    {
        dhHbnum.startFrame(frnr, fr.time);
    }
    if (!fnmHbdistOut_.empty())
    {
        dhDist.startFrame(frnr, fr.time);
    }
    if (!fnmHbangOut_.empty())
    {
        dhAng.startFrame(frnr, fr.time);
    }
    if (!fnmHbdanOut_.empty())
    {
        dhDan.startFrame(frnr, fr.time);
    }


    std::vector<HBond>   daMap;
    AnalysisNeighborhood nb;
    nb.setCutoff(cutoff_);
    const t_info* infoTool1 = &refInfo;
    const t_info* infoTool2;
    if (isTwoDiffGroups_)
    {
        infoTool2 = &targetInfo;
    }
    else
    {
        infoTool2 = &refInfo;
    }
    for (std::size_t cycleIterator = 0; cycleIterator < 2; ++cycleIterator)
    {
        std::vector<gmx::RVec> positionsDonor, positionsAcceptor;
        for (const auto& acceptor : infoTool1->acceptors)
        {
            positionsAcceptor.emplace_back(fr.x[acceptor.ai]);
        }
        for (const auto& donor : infoTool2->donors)
        {
            positionsDonor.emplace_back(fr.x[donor.ai]);
        }
        AnalysisNeighborhoodPositions       nbPos(positionsAcceptor);
        gmx::AnalysisNeighborhoodSearch     start      = nb.initSearch(pbc, nbPos);
        gmx::AnalysisNeighborhoodPairSearch pairSearch = start.startPairSearch(positionsDonor);
        gmx::AnalysisNeighborhoodPair       pair;
        while (pairSearch.findNextPair(&pair))
        {
            if (infoTool1->acceptors[pair.refIndex()].ai == infoTool2->donors[pair.testIndex()].ai)
            {
                continue;
            }
            t_acceptor acceptor = infoTool1->acceptors[pair.refIndex()];
            t_donor    donor    = infoTool2->donors[pair.testIndex()];
            gmx::RVec  vectorDA = { 0, 0, 0 };
            pbc_dx(pbc, fr.x[acceptor.ai], fr.x[donor.ai], vectorDA.as_vec());
            if (vectorDA.norm() > c_rMaxNM_)
            {
                continue;
            }
            else
            {
                for (const auto hIndex : donor.h_atoms)
                {
                    gmx::RVec vectorDH = { 0, 0, 0 };
                    float     degree   = 0;
                    pbc_dx(pbc, fr.x[hIndex], fr.x[donor.ai], vectorDH.as_vec());
                    degree = gmx_angle(vectorDA, vectorDH) * gmx::c_rad2Deg;
                    if (degree < c_angleMaxDegree_)
                    {
                        if (!isTwoDiffGroups_ && (acceptor.isAlsoDonor && acceptor.ai < donor.ai)
                            && mergeHydrogens_)
                        {
                            daMap.emplace_back(acceptor.ai, donor.ai);
                        }
                        else if (mergeHydrogens_)
                        {
                            daMap.emplace_back(donor.ai, acceptor.ai);
                        }
                        else
                        {
                            daMap.emplace_back(donor.ai, acceptor.ai, hIndex);
                        }

                        if (!fnmHbdistOut_.empty())
                        {
                            dhDist.setPoint(distIterator++, vectorDA.norm());
                        }

                        if (!fnmHbangOut_.empty())
                        {
                            dhAng.setPoint(angleIterator++, degree);
                        }
                        if (!fnmHbdanOut_.empty())
                        {
                            if (!isTwoDiffGroups_ && (acceptor.isAlsoDonor && acceptor.ai < donor.ai)
                                && mergeHydrogens_)
                            {
                                donors.insert(acceptor.ai);
                                acceptors.insert(donor.ai);
                            }
                            else
                            {
                                donors.insert(donor.ai);
                                acceptors.insert(acceptor.ai);
                            }
                        }
                    }
                }
            }
        }
        if (isTwoDiffGroups_)
        {
            std::swap(infoTool1, infoTool2);
        }
        else
        {
            break;
        }
    }
    storage_.addData(frnr, daMap);

    if (!fnmHbnumOut_.empty())
    {

        dhHbnum.setPoint(0, prepareFrameData(daMap).size());
        dhHbnum.finishFrame();
    }
    if (!fnmHbdistOut_.empty())
    {
        dhDist.finishFrame();
    }
    if (!fnmHbangOut_.empty())
    {
        dhAng.finishFrame();
    }
    if (!fnmHbdanOut_.empty())
    {
        dhDan.setPoint(0, donors.size());
        dhDan.setPoint(1, acceptors.size());
        dhDan.finishFrame();
    }
}


void Hbond::finishAnalysis(int /*nframes*/)
{
    if (!fnmHbdistOut_.empty())
    {
        AbstractAverageHistogram& averageHistogramDist = histogramModuleDist_->averager();
        averageHistogramDist.normalizeProbability();
        averageHistogramDist.done();
    }
    if (!fnmHbangOut_.empty())
    {
        AbstractAverageHistogram& averageHistogramAng = histogramModuleAng_->averager();
        averageHistogramAng.normalizeProbability();
        averageHistogramAng.done();
    }
}

std::set<HBond> Hbond::prepareFrameData(const std::vector<HbondStorageFrame>& data) const
{
    std::set<HBond> temp;
    for (const auto& i : data)
    {
        for (const auto& j : i.hbondData_)
        {
            temp.insert(j);
        }
    }

    // Detect data duplication and merge it.
    if (mergeHydrogens_)
    {
        for (auto i = temp.begin(); std::next(i, 1) != temp.end(); ++i)
        {
            if (auto j = temp.find({ i->acceptor, i->donor, -1 }); j != temp.end())
            {
                j = std::prev(temp.erase(j), 1);
            }
        }
    }
    return temp;
}

std::set<HBond> Hbond::prepareFrameData(const std::vector<HBond>& data) const
{
    std::set<HBond> temp(data.begin(), data.end());
    // Detect data duplication and merge it.
    if (mergeHydrogens_)
    {
        for (auto i = temp.begin(); std::next(i, 1) != temp.end(); ++i)
        {
            if (auto j = temp.find({ i->acceptor, i->donor, -1 }); j != temp.end())
            {
                j = std::prev(temp.erase(j), 1);
            }
        }
    }
    return temp;
}

void Hbond::writeOutput()
{
    std::vector<HbondStorageFrame> dataOut;
    FILE*                          fp;
    fp      = gmx_ffopen(fnmHbondOut_, "w");
    dataOut = storage_.getData();
    for (const auto& selGroup : { refInfo, targetInfo })
    {
        // Print Selection Indices
        fprintf(fp, "[ %s ]", selGroup.selPtr->name());
        for (std::size_t i = 0; i < selGroup.selPtr->atomIndices().size(); ++i)
        {
            fprintf(fp, "%c %4i", (i % 15 == 0 ? '\n' : ' '), selGroup.selPtr->atomIndices()[i] + 1);
        }
        // Print Donors Indicies
        fprintf(fp, "\n[ donors_hydrogens_%s ]", selGroup.selPtr->name());
        for (const auto& donor : selGroup.donors)
        {
            fprintf(fp, "\n %4i", donor.ai + 1);
            for (std::size_t j = 0; j < donor.h_atoms.size(); ++j)
            {
                fprintf(fp, " %4i", donor.h_atoms[j] + 1);
            }
        }
        // Print Acceptors Indicies
        fprintf(fp, "\n[ acceptors_%s ]", selGroup.selPtr->name());
        for (std::size_t i = 0; i < selGroup.acceptors.size(); ++i)
        {
            fprintf(fp, "%c %4i", (i % 15 == 0 ? '\n' : ' '), selGroup.acceptors[i].ai + 1);
        }
        fprintf(fp, "\n");
        if (!isTwoDiffGroups_)
        {
            break;
        }
    }
    // Print Hbond Indicies
    std::string selects;
    if (isTwoDiffGroups_)
    {
        selects = std::string(refInfo.selPtr->name()) + "-" + std::string(targetInfo.selPtr->name());
    }
    else
    {
        selects = refInfo.selPtr->name();
    }
    if (perFrame_)
    {
        for (const auto& i : dataOut)
        {
            std::set<HBond> tempFrameData = prepareFrameData(i.hbondData_);
            fprintf(fp, "[ hbonds_%s_frame_№%i ]", selects.c_str(), i.frameNumber_);
            for (const auto& j : tempFrameData)
            {
                if (mergeHydrogens_)
                {
                    fprintf(fp, "\n %6i %6i", j.donor + 1, j.acceptor + 1);
                }
                else
                {
                    fprintf(fp, "\n %6i %6i %6i", j.donor + 1, j.hydrogen + 1, j.acceptor + 1);
                }
            }
            fprintf(fp, "\n");
        }
    }
    else
    {
        std::set<HBond> mergedFrameData = prepareFrameData(dataOut);
        fprintf(fp, "[ hbonds_%s ]", selects.c_str());
        if (mergeHydrogens_)
        {
            for (const auto& i : mergedFrameData)
            {
                fprintf(fp, "\n %6i %6i", i.donor + 1, i.acceptor + 1);
            }
        }
        else
        {
            for (const auto& i : mergedFrameData)
            {
                fprintf(fp, "\n %6i %6i %6i", i.donor + 1, i.hydrogen + 1, i.acceptor + 1);
            }
        }
    }
    fprintf(fp, "\n");

    gmx_ffclose(fp);
}

/*! \brief
 * The main function for the analysis template.
 */
int main(int argc, char* argv[])
{
    return gmx::TrajectoryAnalysisCommandLineRunner::runAsMain<Hbond>(argc, argv);
}
