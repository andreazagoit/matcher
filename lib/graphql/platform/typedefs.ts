import { gql } from "graphql-tag";

// Define a restricted subset of the schema for the Platform API
export const platformTypeDefs = gql`
  type Query {
    """
    Check API status
    """
    health: String
    
    """
    Get current authenticated user (if token provided)
    """
    me: UserPublic
  }

  type UserPublic {
    id: ID!
    name: String
    image: String
  }
`;
