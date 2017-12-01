public class FactoryWhite : AbstractFactory{

    public override Apartment createApartment(){
        return new WhiteApartment();
    }

    public override House createHouse(){
        return new WhiteHouse();
    }

}
