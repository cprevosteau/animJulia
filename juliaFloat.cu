////////////////////////////////////////////////////////////////////////////
//	Dessin des ensembles de Julia
//	grâce à CUDA
//	avec une précision float
////////////////////////////////////////////////////////////////////////////


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

//Utilisation de SFML pour gérer l'intéraction avec le clavier et afficher les images résultats
#include <SFML\Graphics.hpp>
#include <SFML\Window.hpp>
#include <SFML\System.hpp>

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

////////////////////////////////////////////////////////////////////////////////


////// Création d'une classe pour les nombres complexes/////
	//Classe surtout utilisée avec la version CPU de ce programme
	//mais très peu dans ce rendu final

class juComplex
{
public:
		float r;
		float i;

	 juComplex():r(0),i(0){};
	 juComplex(float a,float b): r(a),i(b){};
	 float module2()
	 {
		return r*r+i*i;
	 }
	 juComplex operator+(juComplex const& a)
		{
			return juComplex(r+a.r,i+a.i);
		}
	juComplex operator*(juComplex const& a)
		{
			return juComplex(r*a.r-i*a.i,i*a.r+r*a.i);
		}

};






//////////////////////GPU PROGRAM

	/////Déclaration  des objets qui seront stockées dans la mémoire constante du GPU

		__constant__ unsigned int dim[2];				// Dimensions de l'image
		__constant__ float param[5];					// Paramètres arguments de la génération des ensembles de Julia :
														// 0 : Partie réelle de la constante complexe
														// 1 : Partie imaginaire de la constate complexe
														// 2 : Position x de l'image
														// 3 : Position y de l'image
														// 4 : Echelle de l'image

		__constant__ unsigned char couleurDessin[3];	// Couleur RGB associé aux points de l'ensemble de Julia
		__constant__ unsigned char couleurFond[3];		// Couleur RGB associé au fond de l'image
		
	////Valeurs de C possibles par défaut (lorsque l'image est réinitiallisée
		const juComplex C[6]={juComplex(-0.7927,0.1609 ),juComplex(0.32,0.043),juComplex(-1.1380,0.2403),juComplex(-0.0986,-0.65186),juComplex(-0.1225,0.7449),juComplex(-0.3380,-0.6230)};



	////Fonctions min et max pour la gestion des couleurs

unsigned char maxCou(int a,int b)
{
	if(a>=b)
		return (unsigned char)a;
	else
		return (unsigned char)b;
}

unsigned char minCou(int a,int  b)
{
	if(a<=b)
		return (unsigned char)a;
	else
		return (unsigned char)b ;
}


	////Fonctions de mise à jour des paramètres selon l'enfoncement des touches

void gpRaff(float* t_param,unsigned char* t_couDes,unsigned char* t_couFond,sf::Time tdif)
{
	int vitesseD(500);			// vitesse de déplacement de l'image pour x et y
	float vitesseZoom(1.07);	// vitesse de zoom ou dézoom
	int vitesseC(10);			// vitesse de changement de la constante complexe
	int vitesseCou(30);			// vitesse de changement de couleur

	//Modification des positions x et y en fonction de l'échelle et du temps écoulé depuis la dernière boucle tdif

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			{
				t_param[2]-=(tdif.asSeconds())*vitesseD*t_param[4];
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			{
				t_param[2]+=(tdif.asSeconds())*vitesseD*t_param[4];
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
			{
				t_param[3]-=(tdif.asSeconds())*vitesseD*t_param[4];
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
			{
				t_param[3]+=(tdif.asSeconds())*vitesseD*t_param[4];
			}

	//Modification de l'échelle

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Numpad8)||sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
			{
				t_param[4]/=vitesseZoom;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Numpad2)||sf::Keyboard::isKeyPressed(sf::Keyboard::LControl))
			{
				t_param[4]*=vitesseZoom;
			}

	// Modification de la constante complexe en fonction de l'échelle et du temps écoulé tdif

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
			{
				if(t_param[0]<2)
					t_param[0]+=(tdif.asSeconds())*t_param[4]*vitesseC;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			{
				if(t_param[0]>-2)
					t_param[0]-=(tdif.asSeconds())*t_param[4]*vitesseC;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
			{
				if(t_param[1]<2)
					t_param[1]+=(tdif.asSeconds())*t_param[4]*vitesseC;
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			{
				if(t_param[1]>-2)
					t_param[1]-=(tdif.asSeconds())*t_param[4]*vitesseC;
			}


	// Modification de la couleur de dessin

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::R))
			{
				if(t_couDes[0]<255)
					t_couDes[0]=minCou(255,ceil(t_couDes[0]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::F))
			{
				if(t_couDes[0]>0)
					t_couDes[0]=maxCou(0,trunc(t_couDes[0]-tdif.asSeconds()*vitesseCou));
			
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::T))
			{
				if(t_couDes[1]<255)
					t_couDes[1]=minCou(255,ceil(t_couDes[1]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::G))
			{
				if(t_couDes[1]>0)
					t_couDes[1]=maxCou(0,trunc(t_couDes[1]-tdif.asSeconds()*vitesseCou));
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Y))
			{
				if(t_couDes[2]<255)
					t_couDes[2]=minCou(255,ceil(t_couDes[2]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::H))
			{
				if(t_couDes[2]>0)
					t_couDes[2]=maxCou(0,trunc(t_couDes[2]-tdif.asSeconds()*vitesseCou));
			}


	// Modification de la couleur de fond
	
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::U))
			{
				if(t_couFond[0]<255)
					t_couFond[0]=minCou(255,ceil(t_couFond[0]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::J))
			{
				if(t_couFond[0]>0)
					t_couFond[0]=maxCou(0,trunc(t_couFond[0]-tdif.asSeconds()*vitesseCou));
			
			}

				if (sf::Keyboard::isKeyPressed(sf::Keyboard::I))
			{
				if(t_couFond[1]<255)
					t_couFond[1]=minCou(255,ceil(t_couFond[1]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
			{
				if(t_couFond[1]>0)
					t_couFond[1]=maxCou(0,trunc(t_couFond[1]-tdif.asSeconds()*vitesseCou));
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::O))
			{
				if(t_couFond[2]<255)
					t_couFond[2]=minCou(255,ceil(t_couFond[2]+tdif.asSeconds()*vitesseCou));
			}

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))
			{
				if(t_couFond[2]>0)
					t_couFond[2]=maxCou(0,trunc(t_couFond[2]-tdif.asSeconds()*vitesseCou));
			}

}


//////// Fonction exécutée sur le GPU qui renvoie l'intensité de la couleur
	  // en fonction des coordonnées (x,y) du complexe considéré

__device__ float gpJuCouleur(float const& x, float const& y)
{
	float zr(x);
	float zi(y);
	float zr0;
	int intens(0);

		while((zr*zr+zi*zi)<4 && intens<255) // Tant que le module carré de Z est inférieur à 4
		{									 // avec une intensité limitée à 255
			zr0=zr;
			zr=zr*zr-zi*zi+param[0];		// Z reçoit Z^2 + C
			zi=2*zr0*zi+param[1];
			intens++;						// On ajoute de l'intensité

		}
	return (float)(intens)/255.;
}


//////// Fonction kernel qui prend en argument le tableau de pixels stockés sur le GPU
	  // et qui le remplit avec les valeurs associées aux paramètres et 
	  // la fonction gpJuCouleur
	  // Chaque thread correspondra au traitement d'un pixel.

__global__ void dessinKernel(sf::Uint8* dev_pTabPix)
{
	//On détermine les coordonnées du pixel
	// en fonction des coordonnées du thread et du block
	// et des dimensions des blocks et de la grille

	const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;

	if(i<dim[0]&&j<dim[1])											// On vérifie qu'on traite un thread correspondant à un pixel.
	{
			float x(param[2]+(i-(float)(dim[0])/2.)*(param[4]));  // On associe les coordonnées d'un complexe au pixel
			float y(param[3]+(j-(float)(dim[1])/2.)*(param[4]));  // selon les paramètres.

			float pui(gpJuCouleur(x,y));							// On récupère l'intensité associé à ce complexe.

			dev_pTabPix[(i+j*dim[0])*4]=round(pui*couleurDessin[0]+(1-pui)*couleurFond[0]);   // On associe la couleur au pixel
			dev_pTabPix[(i+j*dim[0])*4+1]=round(pui*couleurDessin[1]+(1-pui)*couleurFond[1]); // selon la couleurs de dessin
			dev_pTabPix[(i+j*dim[0])*4+2]=round(pui*couleurDessin[2]+(1-pui)*couleurFond[2]); // et celle de fond.
			dev_pTabPix[(i+j*dim[0])*4+3]=255;
	}

}

////////// Fonction qui prend en argument :
		// Un tableau de pixel sur l'hôte (ou "sur le CPU" en opposition à GPU)
		// Le tableau de pixels associé alloué sur le GPU
		// Un tableau de paramètre sur l'hôte
		// Une couleur de dessin
		// Une couleur de fond
		// Une dimension d'image
		// La fonction remplit le tableau de pixels hôte avec ses nouvelles valeurs.

void gpDessiner(sf::Uint8* pTabPix,sf::Uint8* dev_pixels,float *cpu_param,unsigned char *cpu_couDes,unsigned char *cpu_couFond, unsigned int* cpu_dim)
{
	
	//// Mise à jour des variables constantes (sur le GPU) à partir de leurs homologues hôtes
		
		cudaMemcpyToSymbol(dim, cpu_dim, 2*sizeof(unsigned int));
		cudaMemcpyToSymbol(param, cpu_param, 5*sizeof(float));
		cudaMemcpyToSymbol(couleurDessin, cpu_couDes, 3*sizeof(unsigned char));
		cudaMemcpyToSymbol(couleurFond, cpu_couFond, 3*sizeof(unsigned char));

	//// Définition des dimensions de la grille et des blocks de thread
		
		dim3 grille((cpu_dim[0]+15)/16,(cpu_dim[1]+15)/16); // Assure qu'il y ait au moins assez de thread pour traiter
		dim3 block(16,16);									// tous les pixels.


	// Appel à la fonction noyau qui va remplir le tableau de pixels GPU

		dessinKernel<<<grille,block>>>(dev_pixels);			


	// Mise à jour du tableau de pixels hôtes en fonction de celui GPU

		cudaMemcpy( pTabPix, dev_pixels, cpu_dim[0]*cpu_dim[1] * 4*sizeof(sf::Uint8) , cudaMemcpyDeviceToHost); 
}



int main()
{
	


	// Initiallisation

		bool active(true);		// Est-ce que la fenètre est active ?
		bool ecriture(true);	// Ecriture dans la fenètre (framerate et constante complexe utilisée) ?

		sf::VideoMode video;			// Récupère la résolution du bureau
		video=video.getDesktopMode();

		// On demande l'affichage voulu à l'utilisateur

		char choix;

		std::cout<<"Quelle résolution voulez-vous ?"<<std::endl;
		std::cout<< "Rentrer R pour ensuite indiquer la résolution, sinon rentrer n'importe quoi d'autre."<<std::endl;
		std::cin>>choix;

		bool pleinEcran(false);

		if(choix==char('R')||choix==char('r'))
			{
				int largeur;
				std::cout<<"Indiquer la largeur."<<std::endl;
				std::cin>>largeur;
				int hauteur;
				std::cout<<"Indiquer la hauteur."<<std::endl;
				std::cin>>hauteur;
				video.width=min(largeur,video.width);
				video.height=min(hauteur,video.height);
			}
			else
				pleinEcran=true;




		unsigned int* cpu_dim=new unsigned int[2]; //Dimensions de la fenètre (ou de l'image)
		cpu_dim[0]=video.width;					
		cpu_dim[1]=video.height;

		sf::Texture texture;						//Texture de l'image
		texture.create(cpu_dim[0],cpu_dim[1]);		// avec les bonnes dimensions
		
		float* t_param =new float[5];				// Initiallisation du vecteur de paramètres
		t_param[0]=-0.7927;
		t_param[1]=0.1609;
		t_param[2]=0;
		t_param[3]=0;
		t_param[4]=0.01;

		unsigned char* t_couDes=new unsigned char[3]; // Initiallisation de la couleur de dessin
		t_couDes[0]=255;
		t_couDes[1]=255;
		t_couDes[2]=255;

		unsigned char* t_couFond=new unsigned char[3]; // Initiallisation de la couleur de fond
		t_couFond[0]=0;
		t_couFond[1]=0;
		t_couFond[2]=0;

		sf::Uint8* pixels = new sf::Uint8[cpu_dim[0]*cpu_dim[1] * 4];					// Allocation d'un tableau de pixels
		sf::Uint8* dev_pixels;															// Allocation de son homologue GPU
		cudaMalloc( (void**)&dev_pixels, cpu_dim[0]*cpu_dim[1] * 4*sizeof(sf::Uint8) );

		gpDessiner(pixels,dev_pixels,t_param,t_couDes,t_couFond,cpu_dim );				// Mise à jour du tableau de pixels

	
		texture.update(pixels);								// Mise à jour de la texture 												
		sf::Sprite sprite;									// Création de l'objet associé qui sera affiché dans la fenètre
		sprite.setTexture(texture);
		sf::RenderWindow window;
		if(pleinEcran)
			window.create(video, "Ensemble de Julia",sf::Style::Fullscreen); //Création de la fenètre
		else
			window.create(video, "Ensemble de Julia"); 

		window.setVerticalSyncEnabled(true);
		window.draw(sprite);											   //Desin de l'objet dans la fenètre
		window.display();												   //Mise à jour de l'affichage de la fenètre

	


		sf::Clock horloge;						//Initiallisation de l'horloge
		sf::Time t(horloge.getElapsedTime());	//Initiallisation des variables temps
		sf::Time t1;
		sf::Time tdif;

		sf::Text text;							//Initiallisation du texte à afficher dans la fenètre
		sf::Font font;
		font.loadFromFile("arial.ttf");
		text.setFont(font);
		text.setCharacterSize(24);
		text.setColor(sf::Color::Color(255-t_couFond[0],255-t_couFond[1],255-t_couFond[2])); // Texte qui aura une couleur opposée à celle du fond

	
// On fait tourner le programme jusqu'à ce que la fenêtre soit fermée

		while (window.isOpen())
		{
		
			// On inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
			
			sf::Event event;
			while (window.pollEvent(event) || !active)
			{
			
			/* Tentative d'adapter la résolution en fonction de l'agrandissement/réduction de la fenêtre

				if (event.type == sf::Event::Resized)
				{
					
					cpu_dim[0]=event.size.width;
					cpu_dim[1]=event.size.height;

					delete pixels;
					cudaFree( dev_pixels);
					pixels = new sf::Uint8[cpu_dim[0]*cpu_dim[1] * 4];
					cudaMalloc( (void**)&dev_pixels, cpu_dim[0]*cpu_dim[1] * 4*sizeof(sf::Uint8) );
					texture.create(cpu_dim[0],cpu_dim[1]);
					window.create(sf::VideoMode(cpu_dim[0],cpu_dim[1]),"Ensemble de Julia");

				}
			*/

			// On détermine si la fenêtre est active

				if (event.type == sf::Event::LostFocus)
						active=false;

				if (event.type == sf::Event::GainedFocus)
						active=true;


				// Evènement "fermeture demandée" : on ferme la fenêtre

				if (event.type == sf::Event::Closed)
					window.close();

				if (event.type ==sf::Event::KeyPressed && active) // Si une touche est pressée et que la fenètre est active
					{	
						switch(event.key.code)
						{
						case sf::Keyboard::A:					//Switch entre ecriture ou non du texte
							ecriture=!ecriture;
							break;
							
						case sf::Keyboard::Escape:				// On ferme la fenètre
							window.close();
							break;

						case sf::Keyboard::Space:				//On réinitiallise la constante complexe et les couleurs
						{
								juComplex nC(C[rand()%6]);
								t_param[0]=nC.r;
								t_param[1]=nC.i;

								t_couFond[0]=(rand())%256;
								t_couFond[1]=(rand())%256;
								t_couFond[2]=(rand())%256;

								t_couDes[0]=(rand())%256;
								t_couDes[1]=(rand())%256;
								t_couDes[2]=(rand())%256;
						
								break;
						}
						default:
							break;
						}
					}
			}

			if(active)
			{
				//On récupère le temps écoulé depuis la dernière itération
					t1 = horloge.getElapsedTime();
					tdif=t1-t;
					t=t1;

				//On met à jour les paramètres
					gpRaff(t_param,t_couDes,t_couFond,tdif);

				//On met à jour les pixels et la fenètres
					gpDessiner(pixels,dev_pixels,t_param,t_couDes,t_couFond,cpu_dim );
					texture.update(pixels);
					texture.setSmooth(true);
					window.clear();
					window.draw(sprite);

				// Si écriture, on affiche le framerate et la constante complexe utilisée
					if(ecriture)
					{
						text.setColor(sf::Color::Color(255-t_couFond[0],255-t_couFond[1],255-t_couFond[2]));
						text.setString(std::to_string(1/tdif.asSeconds())+" "+std::to_string(t_param[0])+"+"+std::to_string(t_param[1])+"i");
						window.draw(text);
					}

				//Raffraichissement de la fenètre
					window.display();
			}
		}

		//Désallocation

		delete pixels;
		cudaFree( dev_pixels);

	

    return 0;


}




