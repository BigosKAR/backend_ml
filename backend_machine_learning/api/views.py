# views.py
from rest_framework.response import Response
from rest_framework.decorators import api_view
from models.price_prediction import pricePrediction

@api_view(['POST'])  # POST only is more secure
def getData(request):
    # Example: expects JSON like {"features": [1500, 20, 210, 16, 14, 94, 1]}
    input_data = request.data.get("features", [])

    if not input_data or not isinstance(input_data, list):
        return Response({"error": "Invalid input. Provide a list of features."}, status=400)

    try:
        predicted_price = pricePrediction(input_data)

        return Response({
            'predicted_price': predicted_price
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
