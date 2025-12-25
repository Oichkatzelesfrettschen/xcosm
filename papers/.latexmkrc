# ============================================================================
# COSMOS Papers - latexmk configuration
# ============================================================================
# Shared settings for all papers in the COSMOS collection

# Use pdflatex
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';

# Use bibtex for bibliography
$bibtex_use = 2;

# Clean up auxiliary files
$clean_ext = 'aux bbl blg log out toc lof lot fls fdb_latexmk synctex.gz nav snm vrb run.xml %R-blx.bib';

# Don't ask for user input
$pdflatex_silent_switch = '-interaction=batchmode';

# Output directory (keep files organized)
# $out_dir = 'build';

# Preview mode settings
$preview_continuous_mode = 1;
$pdf_previewer = 'open -a Preview %O %S';  # macOS; use 'evince' on Linux
