public class FactoryBlack : AbstractFactory{

    public override Apartment createApartment(){
        return new BlackApartment();
    }

    public override House createHouse(){
        return new BlackHouse();
    }

}
