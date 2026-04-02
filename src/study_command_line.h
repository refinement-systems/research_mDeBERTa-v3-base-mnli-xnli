#pragma once

#include "optparse.h"
#include "study_types.h"

#include <string>

namespace nli {

std::string DefaultStudyScratchpadRoot();
std::string DefaultStudyCatalogPath();

void ConfigureStudyInitOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildStudyInitOptionParser();
StudyInitCommandLineOptions FinalizeStudyInitCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);

void ConfigureStudyRunOptionParser(optparse::OptionParser& parser);
optparse::OptionParser BuildStudyRunOptionParser();
StudyRunCommandLineOptions FinalizeStudyRunCommandLine(
    const optparse::OptionParser& parser,
    const optparse::Values& options);

}  // namespace nli

