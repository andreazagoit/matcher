import { gql } from "graphql-tag";
import { USER_FRAGMENT } from "@/lib/models/users/gql";

export const GET_DAILY_MATCHES = gql`
  ${USER_FRAGMENT}
  query GetDailyMatches {
    dailyMatches {
      ...UserFields
    }
  }
`;
