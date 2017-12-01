public class FactoryWhite extends AbstractFactory{

    public Apartment createApartment(){
        return new WhiteApartment();
    }

    public House createHouse(){
        return new WhiteHouse();
    }

}
