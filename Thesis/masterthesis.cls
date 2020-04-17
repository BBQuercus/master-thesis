%-----------------------------------------------------%
% Theme largely inspired/adapted from:                %
% @author: Tony Beltramelli - www.tonybeltramelli.com %
%       IT University of Copenhagen - www.itu.dk      %
%       creation date: 2015/06/11, version: 0.1       %
%       last update: 2015/12/05, version: 0.2         %
% Thanks!                                             %
%-----------------------------------------------------%

% Setup
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{masterthesis}[2020/04/01 v0.1 Master's thesis]
\LoadClass[12pt, oneside, letterpaper]{book}

% Basic packages
\RequirePackage[english]{babel}
\RequirePackage[labelfont=bf]{caption}  % caption design
\RequirePackage[table]{xcolor}  % cell colors
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsfonts}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{appendix}  % removed [toc,page]
\RequirePackage{booktabs, multirow}  % borders / merged ranges
\RequirePackage{comment}
\RequirePackage{float}
\RequirePackage{graphicx}
\RequirePackage{indentfirst}  % indents the first paragraph / section
\RequirePackage{listings}
\RequirePackage{longtable}  % multi-page tables static
\RequirePackage{ltablex}  % multi-page tables dynamic
\RequirePackage{soul}  % underlines
\RequirePackage{subcaption}
\RequirePackage{tabularx}  % linebreak in tables
\RequirePackage{textgreek}
\RequirePackage{url}
\RequirePackage{verbatim}
\RequirePackage{mathabx}

% Colors
\RequirePackage{color}
\definecolor{special}{rgb}{0.3, 0.3, 0.3}

% Headers / footers
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}  % removes header line
\fancyhf{}
\lhead{\slshape\nouppercase{\leftmark}}
\cfoot{\thepage}

% Links
\RequirePackage{hyperref}
\hypersetup{
	linktocpage,
    colorlinks,
    citecolor=special,
    filecolor=black,
    linkcolor=black,
    urlcolor=special,
}

% Pagination
\expandafter\def\expandafter\normalsize\expandafter{%
    \normalsize
    \setlength\abovedisplayskip{0pt}
    \setlength\belowdisplayskip{3pt}
    \setlength\abovedisplayshortskip{0pt}
    \setlength\belowdisplayshortskip{3pt}
}

% References
\renewcommand\bibname{References}

% SyncTex synchronization
\RequirePackage{pdfsync}
\synctex=1

% Font styling
\usepackage[scaled]{helvet}
\renewcommand\familydefault{\sfdefault} 
\usepackage[T1]{fontenc}

% Chapter heading styling
\RequirePackage[helvetica]{quotchap}
\definecolor{chaptergrey}{rgb}{0.5,0.5,0.5}

% Line spacing
\RequirePackage{setspace} 
\onehalfspacing

% Table styling
\RequirePackage{multirow}

% Paragraph styling
\parindent 12pt
\parskip 1em

% Sectioning
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

% Abstract
\newcommand{\abstractpage}{
	\newpage
	\chapter*{Abstract}
	\input{frontmatter/abstract}
	\addcontentsline{toc}{chapter}{\numberline{}Abstract}
}

% Acknowledgements
\newcommand{\acknowledgments}{
    \newpage
	\chapter*{Acknowledgments}
	\input{frontmatter/acknowledgments}
	\addcontentsline{toc}{chapter}{\numberline{}Acknowledgments}
}

% TOC
\newcommand{\contents}{
    \newpage
    \addcontentsline{toc}{chapter}{\numberline{}Contents}
    \setlength{\parskip}{0pt}
    \tableofcontents
}

% List Figures
\newcommand{\listfigures}{
    \newpage
    \addcontentsline{toc}{chapter}{\numberline{}List of figures}
    \listoffigures
}

% List Tables
\newcommand{\listtables}{
    \newpage
    \addcontentsline{toc}{chapter}{\numberline{}List of tables}
    \setlength{\parskip}{1em}
    \listoftables
}

% Reagent tables
\newcommand{\regtable}[1]{
    \caption{#1}\\\toprule\endfirsthead
    \toprule\textbf{Reagent or Resource} & \textbf{Source} & \textbf{Identifier} \\\midrule\endhead
    \midrule\multicolumn{3}{r}{\itshape continues on next page}\\\midrule\endfoot
    \bottomrule\endlastfoot
    \textbf{Reagent or Resource} & \textbf{Source} & \textbf{Identifier} \\\midrule
}

% Chapter headlines
\newcommand{\chapterheadline}[1]{
    \renewcommand\chapterheadstartvskip{\vspace*{#1\baselineskip}}
}

% Bibliograpy
\newcommand{\makebibliography}{
    \newpage
    \chapterheadline{-4}
    
    \addcontentsline{toc}{chapter}{\numberline{}Bibliography}
    \bibliography{references}
    \bibliographystyle{unsrt}
    
    \chapterheadline{-8}
}

% Front matter / sectioning
\newcommand{\makefrontmatter}{
    \setcounter{page}{1}
    \pagenumbering{roman}
    
    \chapterheadline{-4}
    
    \abstractpage
    \acknowledgments
    \contents
    % \listfigures
    % \listtables
    
    \clearpage
    
    \setcounter{page}{1}
    \pagenumbering{arabic}
    
    \chapterheadline{-8}
}

% Colophon
\AtEndDocument{%
\thispagestyle{plain}
\begin{flushright}
    \section*{Colophon}
    This document was created using \LaTeX{} and Bib\TeX{} typesetting originally developed by Leslie Lamport,
        based on \TeX{} created by Donald Knuth. \\
    Theme adapted from a design by Tony Beltramelli. \\
    Edited in Visual Studio Code. \\
    The text is set in Helvetica.
\end{flushright}}

% Title page
\def\degree#1{\gdef\@degree{#1}}
\def\university#1{\gdef\@university{#1}}
\def\location#1{\gdef\@location{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}

\renewcommand{\maketitle}{
    \begin{titlepage}
        \begin{center}
            \setlength{\parskip}{0pt}
            
            \begin{flushright}
                \includegraphics[height=2cm]{images/fmi_logo.jpg}
            \end{flushright}
            
            \vspace*{10mm}
            
            {\Large {Master Thesis} \par}
            \vspace*{5mm}
            
            {\huge {\bfseries \@title} \par}
            
            \vspace*{10mm}
            
            {\large {\textit{Author:} \@author} \par}
            {\large {\textit{Supervisor:} \@supervisor} \par}
            
            \vspace*{10mm}
            
            {\large \textsc{\@university} \\ \@location \par}
            
            \vspace*{10mm}
            
            {\large\it {\@degree} \par}
            
            \vspace*{5mm}
            
            {\large \@date \par}
        \end{center}
    \end{titlepage}
}
