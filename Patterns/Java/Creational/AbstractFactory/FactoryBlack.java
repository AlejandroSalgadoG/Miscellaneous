public class FactoryBlack extends AbstractFactory{

    public Apartment createApartment(){
        return new BlackApartment();
    }

    public House createHouse(){
        return new BlackHouse();
    }

}
