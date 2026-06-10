\LoadClass[a4paper,11pt]{article} % Classe du document

%---------- Packages ----------------

\RequirePackage[english]{babel} %Langue du document
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage[T1]{fontenc} %Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=2cm,bottom=3.5cm]{geometry} %Configuration de la page

\usepackage{sectsty}
\usepackage{xcolor}

% On définit les deux couleurs utilisées
\definecolor{couleur_section}{RGB}{0,0,128}
\definecolor{couleur_subsection}{RGB}{0,128,255}

% On personnalise chacun des titres à l'aide des commandes "sectionfont", "subsectionfont", etc.
\sectionfont{\color{couleur_section}}
\subsectionfont{\color{couleur_subsection}}
\subsubsectionfont{\itshape}
%------- Informations sur le rapport----------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\enseignantenseirb}[1]{\renewcommand{\enseignantenseirb}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{1cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.15]{logo_enseirb.jpeg}} %Affichage de l'image au top de la page
\rhead{\includegraphics[scale=0.045]{insa_logo.png}}
\chead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}


	\centering %Centraliser le contenu
	\includegraphics[width=0.5\textwidth]{logo_enseirb.jpeg}\par\vspace{1cm} %Insertion du logo
	\includegraphics[width=0.5\textwidth]{insa_logo.png}\par\vspace{1cm} %Insertion du logo
	{\scshape\Large \UE \\ \sujet \\ \par} %sous-titre
	\vspace{1cm}%Espace de 1cm
    \rule{\linewidth}{0.4 mm} \\[0.4 cm]
	{\huge\bfseries \titre \par}
    \rule{\linewidth}{0.4 mm} \\[0.0 cm]
	\vspace{1cm}%Espace de 3cm
    
	\begin{minipage}{0.4\textwidth} %Minipage pour faire deux colonnes
		\begin{flushleft} \large %Envoyer à gauche
		\emph{\textbf{Auteur :}}\\ %Pour le titre au dessus des noms à gauche
        \eleves
        \vspace{1cm}%Espace de 1cm
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Sous la supervision de :}}\\
		 \enseignant 
		\end{flushright}
	\end{minipage}\\[2cm]
    
	\vfill
	{Année 2025-2026} %Affichage de la date

\end{titlepage}
}

\newcommand{\tabledematieres}{

\tableofcontents
\newpage

}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}