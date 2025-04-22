from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime

# ---------------------------------------------
# Function to display the user menu
# ---------------------------------------------
def menu() -> int:
    print("********************************************************")
    print("|  1: Total sales per month/year                       |")
    print("|  2: Online vs in-store orders                        |")
    print("|  3: Top-selling products                             |")
    print("|  4: Average order value over time                    |")
    print("|  5: Most popular payment methods                     |")
    print("|  0: Exit                                             |")
    print("********************************************************")
    try:
        return int(input("Enter your choice: "))
    except ValueError:
        return -1

# ---------------------------------------------
# Q1: Show total sales grouped by year and month
# ---------------------------------------------
def q1_total_sales_by_month(df):
    print("Running Q1: Monthly Sales Trend...")
    result = df.withColumn("year", F.year("order_date")) \
               .withColumn("month", F.month("order_date")) \
               .groupBy("year", "month") \
               .agg(F.sum("total_amount").alias("total_sales")) \
               .orderBy("year", "month")
    result.show(20)

# ---------------------------------------------
# Q2: Compare number of online vs in-store orders
# ---------------------------------------------
def q2_online_vs_store(df):
    print("Running Q2: Online vs In-store Orders...")
    result = df.groupBy("is_online") \
               .agg(F.countDistinct("order_id").alias("total_orders")) \
               .orderBy("is_online")
    result.show()

# ---------------------------------------------
# Q3: Top 10 products by quantity sold
# ---------------------------------------------
def q3_top_selling_products(df):
    print("Running Q3: Top Selling Products...")
    result = df.groupBy("product_name") \
               .agg(F.sum("quantity").alias("total_quantity")) \
               .orderBy(F.desc("total_quantity")) \
               .limit(10)
    result.show()

# ---------------------------------------------
# Q4: Average order value per month/year
# ---------------------------------------------
def q4_avg_order_value(df):
    print("Running Q4: Average Order Value by Month...")
    result = df.withColumn("year", F.year("order_date")) \
               .withColumn("month", F.month("order_date")) \
               .groupBy("year", "month") \
               .agg(F.avg("total_amount").alias("avg_order_value")) \
               .orderBy("year", "month")
    result.show()

# ---------------------------------------------
# Q5: Most commonly used payment methods
# ---------------------------------------------
def q5_payment_methods(df):
    print("Running Q5: Popular Payment Methods...")
    result = df.groupBy("payment_method") \
               .agg(F.count("order_id").alias("payment_count")) \
               .orderBy(F.desc("payment_count"))
    result.show()

# ---------------------------------------------
# Main function to initialize Spark, load data, and run menu
# ---------------------------------------------
def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("eCommerce Data Analysis") \
        .master("local[*]") \
        .getOrCreate()

    # Suppress Spark logs to show only errors
    spark.sparkContext.setLogLevel("ERROR")

    # Read the TSV data file into a DataFrame
    df = spark.read.option("header", "true") \
                   .option("sep", "\t") \
                   .option("inferSchema", "true") \
                   .csv("data/ecommerce_data.tsv") \
                   .withColumn("order_date", F.to_date("order_date", "yyyy-MM-dd"))

    # Menu loop for user interaction
    while True:
        choice = menu()
        if choice == 1:
            q1_total_sales_by_month(df)
        elif choice == 2:
            q2_online_vs_store(df)
        elif choice == 3:
            q3_top_selling_products(df)
        elif choice == 4:
            q4_avg_order_value(df)
        elif choice == 5:
            q5_payment_methods(df)
        elif choice == 0:
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice, try again.")

# Q6: Revenue by product category
def q6_revenue_by_category(df):
    print("Running Q6: Revenue by Product Category...")
    result = df.groupBy("category") \
               .agg(F.sum("total_amount").alias("category_revenue")) \
               .orderBy(F.desc("category_revenue"))
    result.show()

# Q7: Quantity sold by product category
def q7_quantity_by_category(df):
    print("Running Q7: Quantity Sold by Category...")
    result = df.groupBy("category") \
               .agg(F.sum("quantity").alias("total_quantity")) \
               .orderBy(F.desc("total_quantity"))
    result.show()

# Q8: Repeat vs one-time customers
def q8_repeat_vs_one_time(df):
    print("Running Q8: Repeat vs One-Time Customers...")
    customer_order_counts = df.groupBy("customer_id") \
                              .agg(F.countDistinct("order_id").alias("orders"))
    
    result = customer_order_counts.withColumn("customer_type", 
               F.when(F.col("orders") > 1, "Repeat").otherwise("One-Time")) \
               .groupBy("customer_type") \
               .agg(F.count("*").alias("num_customers"))
    
    result.show()

# Q9: Orders and revenue by location
def q9_orders_revenue_by_location(df):
    print("Running Q9: Orders and Revenue by Location...")
    result = df.groupBy("location") \
               .agg(
                   F.countDistinct("order_id").alias("total_orders"),
                   F.sum("total_amount").alias("total_revenue")
               ).orderBy(F.desc("total_revenue"))
    result.show()

# Q10: Daily sales trend
def q10_daily_sales(df):
    print("Running Q10: Daily Sales Trend...")
    result = df.groupBy("order_date") \
               .agg(F.sum("total_amount").alias("daily_revenue")) \
               .orderBy("order_date")
    result.show(30)  # Display the first 30 days of sales


# ---------------------------------------------
# Entry point of the program
# ---------------------------------------------
if __name__ == "__main__":
    main()
