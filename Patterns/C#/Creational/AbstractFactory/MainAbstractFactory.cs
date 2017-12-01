public class MainAbstractFactory{

    public static void Main(string[] args){

        AbstractFactory factoryWhite = FactoryCreator.createFactoryWhite();
        AbstractFactory factoryBlack = FactoryCreator.createFactoryBlack();

        Apartment whiteApartment = factoryWhite.createApartment();
        Apartment blackApartment = factoryBlack.createApartment();

        House whiteHouse = factoryWhite.createHouse();
        House blackHouse = factoryBlack.createHouse();

        whiteApartment.getApartmentColor();
        blackApartment.getApartmentColor();

        whiteHouse.getHouseColor();
        blackHouse.getHouseColor();

    }

}
