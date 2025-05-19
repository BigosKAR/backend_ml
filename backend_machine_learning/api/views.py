from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from models.price_prediction import predict_price_with_explanation  # Updated import

class PredictPriceView(APIView):
    def post(self, request):
        user_input = request.data  # Expecting a dict (JSON object)

        if not isinstance(user_input, dict):
            return Response(
                {"error": "Invalid input format. Expected a JSON object."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = predict_price_with_explanation(user_input)
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
